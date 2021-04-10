__author__ = "Gurdeep"

from sklearn.model_selection import train_test_split
from LoadDataset import VideoDatasetHandler
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
try:
    import keras
except Exception:
    print("Keras not present, trying Tensorflow")
    try:
        import tensorflow.keras as keras
    except Exception:
        print("Tensorflow not present")
        print("Either install tensorflow or keras")
        exit(0)
import cv2
import os

class PrepareDataset(VideoDatasetHandler):
    def __init__(self, master_directory, prepared_data_directory="PreparedDataset"):
        super(PrepareDataset, self).__init__(master_directory)
        self.data_dir = prepared_data_directory
        self.makeDir()
        self.master_csv_file = self.data_dir+"\\"+"DatasetMasterCsvFile.csv"
        self.train_csv_file = self.data_dir+"\\"+"DatasetTrainCsvFile.csv"
        self.test_csv_file = self.data_dir+"\\"+"DatasetTestCsvFile.csv"
        self.validation_csv_file = self.data_dir+"\\"+"DatasetValidationCsvFile.csv"
        #for keras.utils.to_categorical 
        self.num_classes = 8 
        #for data augmentation
        self.augmentation_object = keras.preprocessing.image.ImageDataGenerator()
        self.brightness_lower_index = 25
        self.brightness_higher_index = 100
        self.rotation_higher_index = 100
        self.rotation_lower_index = 99

    def makeDir(self):
        try:
            os.mkdir(self.data_dir)
            print("\"{}\" Directory created in current directory".format(self.data_dir))
        except Exception:
            print("\nWARNING: \"{}\" Directory Already Exists\n".format(str(Path(self.data_dir).resolve())))
        self.data_dir = str(Path(self.data_dir).resolve())

    def openFile(self, mode):
        if(mode.lower() == 'master'):
            if(os.path.isfile(self.master_csv_file)):
                return pd.read_csv(self.master_csv_file, index_col=0)
            else:
                print("ERROR in PrepareDataset/openFile: master csv file does not exists.")
        elif(mode.lower() == 'train'):
            if(os.path.isfile(self.train_csv_file)):
                return pd.read_csv(self.train_csv_file, index_col=0)
            else:
                print("ERROR in PrepareDataset/openFile: train csv file does not exists.")
        elif(mode.lower() == 'test'):
            if(os.path.isfile(self.test_csv_file)):
                return pd.read_csv(self.test_csv_file, index_col=0)
            else:
                print("ERROR in PrepareDataset/openFile: test csv file does not exists.")
        elif(mode.lower() == 'validation'):
            if(os.path.isfile(self.validation_csv_file)):
                return pd.read_csv(self.validation_csv_file, index_col=0)
            else:
                print("ERROR in PrepareDataset/openFile: validation csv file does not exists.")
        else:
            print("ERROR in PrepareDataset/saveFile: Wrong/incompatible mode type given")
        exit(0)

    def saveFile(self, df, mode):
        if(mode.lower() == 'master'):
            df.to_csv(self.master_csv_file)
            self.master_csv_file = str(Path(self.master_csv_file).resolve())
        elif(mode.lower() == 'train'):
            df.to_csv(self.train_csv_file)
            self.train_csv_file = str(Path(self.train_csv_file).resolve())
        elif(mode.lower() == 'test'):
            df.to_csv(self.test_csv_file)
            self.test_csv_file = str(Path(self.test_csv_file).resolve())
        elif(mode.lower() == 'validation'):
            df.to_csv(self.validation_csv_file)
            self.validation_csv_file = str(Path(self.validation_csv_file).resolve())
        else:
            print("ERROR in PrepareDataset/saveFile: Wrong/incompatible mode type given")
            return False
        return True

    def saveGeneratorInfo(self, gen_kwargs):
        mode = gen_kwargs['mode']
        if(mode == 'None'):
            mode = 'CustomDataframe'
        try:
            with open(self.data_dir+'\\'+mode+"GenratorInfo.json", "w") as f:
                json.dump(gen_kwargs, f)
        except Exception as exx:
            print("Error in PrepareDataset/saveGeneratorInfo:", exx, '\n')
            return False
        return True

    def getFileRows(self, mode, df=None):
        if(mode == None):
            if(type(df) == type(None) and df == None):
                print("ERROR in PreparedDataset/getFileRows: No DataFrame given, it expects it because mode is None/not string")
                exit(0) 
        elif(mode not in ['master', 'test', 'train', 'validation']):
            print("ERROR in PreparedDataset/getFileRows: Wrong/incompatible mode given")
            exit(0)
        else:
            df = self.openFile(mode=mode)

        return int(len(df.index))

    @staticmethod
    def removeDecisions(df, remove_list):
        df=df[~df['decisions'].isin(remove_list)]
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def removeHighestOccurringDecision(df, rows_to_delete, print_it=False):
        if(rows_to_delete <= 0):
            return df
        for i in range(rows_to_delete):
            highest_occuring_label = df.decisions.value_counts().idxmax()
            highest_occuring_label_df = df[df['decisions'] == highest_occuring_label]
            video_file_with_highest_occurence = highest_occuring_label_df.vid_file_path.value_counts().idxmax()
            row_index_to_delete = highest_occuring_label_df[highest_occuring_label_df.vid_file_path == video_file_with_highest_occurence].iloc[0].name
            df.drop(row_index_to_delete, inplace=True)
            if(print_it == True):
                print("Highest Occurring Label is {}, Row with index {} deleted.".format(highest_occuring_label, row_index_to_delete))
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def resize_image(image, shape, print_it=True):
        (h, w) = image.shape[:2]
        if(print_it == True):
            print("WARNING in PrepareDataset/resize_image: Image will loose contents when resizing from original shape")
        if(int(len(shape)) < 2):
            print("ERROR in PrepareDataset/resize_image: Wrong shape given for resizing image")
            exit(0)
        width, height = shape[1], shape[0]
        interpolation = cv2.INTER_AREA
        if(width>w and height>h):
            print("width={}, w={}, height={}, h={}".format(height, h, width, w))
            interpolation=cv2.INTER_CUBIC
        return cv2.resize(image, shape[:2], interpolation=interpolation)

    def createMasterDataframe(self, skip=False, clean_and_cvt=True, remove_cmds=['BRG', 'BLF']):
        df = pd.DataFrame()
        #remove the extra_length variable with a function to find the longest vid_file path and use that instead, this is just temp sol. 
        vid_file_path_len, extra_length, i = None, 16, 0 #extra_length should be increased if no of videos or scene dirs increases 100 
        print("Creating Master Dataset Dataframe ...")
        print("This might take time depending on the size of the dataset")
        while(True):
            frames_ret, frames, decisions, vid_file = self.vidTagGen(gray=True, skip=skip, get_file=True) 
            if(frames_ret == False):
                if(skip == True and vid_file == None):
                    continue
                break

            vid_file_path = str(Path(vid_file).resolve())
            if(i==0):
                vid_file_path_len = int(len(vid_file_path))+extra_length
                i+=1

            sub_df = pd.DataFrame()
            sub_df['vid_file_path'] = np.full(decisions.shape, vid_file_path, dtype="<U{}".format(vid_file_path_len))
            sub_df['decisions'] = decisions
            sub_df['frame_index'] = np.arange(decisions.shape[0], dtype=np.int64)

            df = df.append(sub_df.copy(), ignore_index=True)
        if(remove_cmds!=None):
            total_cmds = np.unique(df.decisions)
            remove_set= set()
            for cmd in total_cmds:
                for r_cmd in remove_cmds:
                    if(r_cmd in cmd):
                        remove_set.add(cmd)
            df = self.removeDecisions(df, list(remove_set))

        if(clean_and_cvt == True):
            df.decisions = self.onlyDecision(df.decisions)
            self.saveObjAsPickle(self.str2IntDict(df.decisions), "str2int_dictionary.pickle")
            df.decisions = self.cvtStr2Int(df.decisions)
        
        return df

    def getChunkOfFile(self, start_index, end_index, mode, df=None):
        if(mode == None):
            if(type(df) == type(None) and df == None):
                print("ERROR in PreparedDataset/getChunkOfFile: No DataFrame given, it expects it because mode is None/not string")
                exit(0) 
        elif(mode not in ['master', 'test', 'train', 'validation']):
            print("ERROR in PreparedDataset/getChunkOfFile: Wrong/incompatible mode given")
            exit(0)
        else:
            df = self.openFile(mode=mode)

        if(start_index < 0 or start_index > int(len(df.index))):
            print("Error in PreparedDataset/getChunkOfFile: start_index out of bound")
            exit(0)
        if(end_index < 0 or end_index > int(len(df.index))):
            print("Error in PreparedDataset/getChunkOfFile: end_index out of bound")
            exit(0)
        return df.iloc[start_index:end_index]

    def setBrightnessRange(self, lower_index, higher_index):
        if(100<lower_index<1):
            print("Error in PreparedDataset/setBrightnessRange: lower_index out of bound, accepted range = 1-100")
            exit(0)
        if(100<higher_index<1):
            print("Error in PreparedDataset/setBrightnessRange: higher_index out of bound, accepted range = 1-100")
            exit(0)
        if(higher_index<=lower_index):
            print("Error in PreparedDataset/setBrightnessRange: higher_index <= lower_index")
            exit(0)
        self.brightness_lower_index = lower_index
        self.brightness_higher_index = higher_index

    def setRotationRange(self, lower_index, higher_index):
        if(higher_index<=lower_index):
            print("Error in PreparedDataset/setRotationRange: higher_index <= lower_index")
            exit(0)
        self.rotation_lower_index = lower_index
        self.rotation_higher_index = higher_index

    def getFrame(self, vid_file, frame_index, gray=False, flatten=False, dtype=None, resize_shape=None, reshape_image=None, augmentation=False):
        cap = cv2.VideoCapture(vid_file)
        if(cap.set(1, frame_index) == False):
            print("ERROR in PrepareDataset/getFrame: Setting video file pointer failed")
            print("vid_file =", vid_file)
            print("frame_index =", frame_index)
            print("total frames =", cv2.get(cv2.CAP_PROP_FRAME_COUNT))
            exit(0)
        ret, frame = cap.read()
        if(ret == False):
            print("ERROR in PrepareDataset/getFrame: Reading video file frame failed")
            print("vid_file =", vid_file)
            print("frame_index =", frame_index)
            print("total frames =", cv2.get(cv2.CAP_PROP_FRAME_COUNT))
            exit(0)
        cap.release()

        if(augmentation == True):
            rotation_list = [np.random.randint(-self.rotation_higher_index, -self.rotation_lower_index)/100, np.random.randint(self.rotation_lower_index, self.rotation_higher_index)/100]
            params =  {'theta': np.random.choice(rotation_list),
                       'brightness': np.random.randint(self.brightness_lower_index, self.brightness_higher_index)/100
                        }
            frame = self.augmentation_object.apply_transform(x=frame, transform_parameters=params).astype(frame.dtype)

        if(gray == True):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if(resize_shape!=None):
            frame = self.resize_image(frame, resize_shape, print_it=False)
        if(flatten == True and reshape_image != None):
            print("Warning: flatten and reshape used together. Make sure you know what you are doing")
        if(flatten == True):
            frame = frame.flatten() 
        if(reshape_image != None):
            frame = frame.reshape(reshape_image)
        if(dtype == None):
            return frame
        return frame.astype(dtype)

    def getFileChunkArray(self, df, gray=False, flatten=False, dtype=None, a=None, b=None, resize_shape=None, reshape_image=None, to_categorical=False, cat_dtype=None, augmentation=False):
        df_len, frames, decisions = int(len(df.index)), None, None
        for vid_file, decision, frame_index, i in zip(df.vid_file_path, df.decisions, df.frame_index, range(df_len)):
            frame = self.getFrame(vid_file, frame_index=frame_index, gray=gray, flatten=flatten, dtype=dtype, resize_shape=resize_shape, reshape_image=reshape_image, augmentation=augmentation)
            if(i==0):
                shape = list()
                shape.append(df_len)
                shape.extend(list(frame.shape))
                frames = np.empty(shape, dtype=frame.dtype)
                if('int' in str(type(decision))):
                    decisions = np.empty((df_len, ), dtype=df.decisions.dtype)
                elif('str' in str(type(decision))):
                    decisions = np.empty((df_len, ), dtype="<U{}".format(int(len(decision))))
                else:
                    print("ERROR in PrepareDataset/getFileChunkArray: DataFrame.decisions have different dtype other than int/uint or str")
                    exit(0)
            frames[i] = frame
            decisions[i] = decision
        if(a!=None):
            frames = self.normalize(frames, a=a, b=b)
        if('str' in str(decisions.dtype) and to_categorical == True):
            print("ERROR in PrepareDataset/getFileChunkArray: Cannot convert to categories when decisions are of string type")
            exit(0)
        elif(to_categorical == True):
            if(cat_dtype == None):
                decisions = keras.utils.to_categorical(decisions, num_classes=self.num_classes).astype(np.uint8)
            else:
                decisions = keras.utils.to_categorical(decisions, num_classes=self.num_classes).astype(cat_dtype)
        return frames, decisions

    def saveObjAsPickle(self, obj, filename):
        path = bytes(self.data_dir+'\\'+filename, 'utf-8')
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    def train_test_split(self, df=None, ret=False, validation=False, test_size=0.15, validation_size=0.15, random_state=None, shuffle=False, stratify=False):
        if(type(df) == type(None) and df == None):
            df = self.openFile(mode='master')
        X, y = df.drop(['decisions'], axis=1), df.drop([col for col in df.keys() if col != 'decisions'], axis=1)

        if('str' in str(type(y.decisions.iloc[0]))):
            y.decisions = self.onlyDecision(y.decisions)
            self.saveObjAsPickle(self.str2IntDict(y.decisions), "str2int_dictionary.pickle")
            y.decisions = self.cvtStr2Int(y.decisions)
        elif('int' not in str(type(y.decisions.iloc[0]))):
            print("ERROR in PrepareDataset/train_test_split: Wrong datatype values present in master dataframe decisions column")
            exit(0)

        total_size = int(len(y.decisions))

        if(stratify == True):
            train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=None, test_size=test_size, random_state=random_state, shuffle=True, stratify=y)
        else:
            train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=None, test_size=test_size, random_state=random_state, shuffle=shuffle)
        
        print("\nSplit Summary:")
        
        if(validation == True):
            val_size = (validation_size*total_size)/int(len(train_y.decisions))
            if(stratify == True):
                train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, train_size=None, test_size=val_size, random_state=random_state, shuffle=True, stratify=train_y)
            else:
                train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, train_size=None, test_size=val_size, random_state=random_state, shuffle=shuffle)
            validation_df = pd.concat([val_x, val_y], axis=1)
            validation_df.reset_index(drop=True, inplace=True)
            self.saveFile(validation_df, mode='validation')
            print("validation size =", len(validation_df.decisions))
            print("percent of validation size, wrt to master file =", (int(len(validation_df.decisions))/total_size)*100, "%")

        train_df = pd.concat([train_x, train_y], axis=1)
        train_df.reset_index(drop=True, inplace=True) 
        test_df = pd.concat([test_x, test_y], axis=1)
        test_df.reset_index(drop=True, inplace=True)
        self.saveFile(train_df, mode='train')
        self.saveFile(test_df, mode='test')

        print("given test and validation sizes =", test_size, validation_size)
        print("test size =", len(test_df.decisions))
        print("percent of test size, wrt to master file =", (int(len(test_df.decisions))/total_size)*100, "%")
        print("train size =", len(train_df.decisions))
        print("percent of train size, wrt to master file =", (int(len(train_df.decisions))/total_size)*100, "%")
                
        if(ret == False):
            return None
        elif(validation == True):
            return train_x, test_x, val_x, train_y, test_y, val_y   
        
        return train_x, test_x, train_y, test_y

    @staticmethod
    def getFactors(no):
        return [n for n in range(1, no+1) if(no%n == 0)]

    def getBatchSizes(self, files=['train', 'test', 'validation', 'master'], df=None):
        batch_size_dict = dict()
        if(df==None):
            for file_mode in files:
                batch_size_dict[file_mode] = self.getFactors(self.getFileRows(mode=file_mode))
        else:
            batch_size_dict['Unknown File'] = self.getFactors(int(len(df.index)))
        return batch_size_dict

    def shapeAccordingToBatchSize(self, batch_size, mode, save=False, df=None, print_it=True):
        if(mode == None):
            if(type(df) == type(None) and df == None):
                print("ERROR in PreparedDataset/shapeAccordingToBatchSize: No DataFrame given, it expects it because mode is None/not string")
                exit(0) 
        elif(mode not in ['master', 'test', 'train', 'validation']):
            print("ERROR in PreparedDataset/shapeAccordingToBatchSize: Wrong/incompatible mode given")
            exit(0)
        else:
            df = self.openFile(mode=mode)

        if(print_it == True):
            print("\nTotal rows deleted to match the batch size =", int(len(df.index))%batch_size, "\n")

        df = self.removeHighestOccurringDecision(df=df, rows_to_delete=int(len(df.index))%batch_size, print_it=False)

        if(save == True):
            if(mode == None):
                print("ERROR in PreparedDataset/shapeAccordingToBatchSize: mode is None, cannot save file")
                exit(0)
            self.saveFile(df, mode=mode)
            return None
        return df

    @staticmethod
    def categorical2Int(cat_array):
        return np.argmax(cat_array, axis=1)

    def stepsPerEpoch(self, mode, batch_size, repeat_factor=1, df=None):
        if(repeat_factor <= 0):
            print("ERROR in PreparedDataset/stepsPerEpoch: repeat_factor <= 0")
            exit(0)
        if(df is not None):
            df = self.shapeAccordingToBatchSize(mode=None, df=df, batch_size=batch_size, print_it=False)
        else:
            df = self.shapeAccordingToBatchSize(mode=mode, batch_size=batch_size, print_it=False)
        steps_per_epoch = self.getFileRows(mode=None, df=df)/batch_size
        return steps_per_epoch*repeat_factor

    def datasetGenerator(self, batch_size, mode, df=None, gray=False, flatten=False, dtype=None, 
                         a=None, b=None, resize_image=None, reshape_image=None, to_categorical=False, 
                         cat_dtype=None, augmentation=False, repeat_factor=1):
        df_val_dict, file_iter = {'df':'None'}, 0
        if(repeat_factor <= 0):
            print("ERROR in PreparedDataset/datasetGenerator: repeat_factor <= 0")
            exit(0)
        if(repeat_factor > 1 and augmentation==False):
            print("ERROR in PreparedDataset/datasetGenerator: repeat_factor > 1 and augmentation is False. Same data will be fed")
            exit(0)
        if(mode == None):
            if(type(df) == type(None) and df == None):
                print("ERROR in PreparedDataset/datasetGenerator: No DataFrame given, it expects it because mode is None/not string")
                exit(0)
            df_val_dict = {'df': 'Dataframe Given'}
        elif(mode not in ['master', 'test', 'train', 'validation']):
            print("ERROR in PreparedDataset/datasetGenerator: Wrong/incompatible mode given")
            exit(0)
        else:
            df = self.openFile(mode=mode)
        
        df = self.shapeAccordingToBatchSize(batch_size=batch_size, mode=None, df=df)
        if('str' in str(type(df.decisions.iloc[0]))):
            print("WARNING in PreparedDataset/datasetGenerator: decisions column in the dataframe is of string type")
            print("Converting them to integers...")
            df.decisions = self.onlyDecision(df.decisions)
            self.saveObjAsPickle(self.str2IntDict(df.decisions), "str2int_dictionary.pickle")
            df.decisions = self.cvtStr2Int(df.decisions)
        elif('int' not in str(type(df.decisions.iloc[0]))):
            print("ERROR in PreparedDataset/datasetGenerator: Wrong datatype values present in dataframe decisions column")
            exit(0)

        df_max_size = self.getFileRows(mode=None, df=df)
        
        kwargs = {'batch_size':str(batch_size), 'mode':str(mode), 'gray':str(gray), 'flatten':str(flatten), 'dtype':str(dtype), 'a':str(a), 'b':str(b), 'resize_image':str(resize_image),
                  'reshape_image':str(reshape_image), 'to_categorical':str(to_categorical), 'cat_dtype':str(cat_dtype), 'file_rows':str(df_max_size)}
        kwargs.update(df_val_dict)
        
        self.saveGeneratorInfo(kwargs)

        while(True):
            if(file_iter+batch_size > df_max_size):
                file_iter = 0
                continue

            chunked_df = self.getChunkOfFile(start_index=file_iter, end_index=file_iter+batch_size, mode=None, df=df)
            frames, decisions = self.getFileChunkArray(df=chunked_df, gray=gray, flatten=flatten, dtype=dtype, a=a, b=b, resize_shape=resize_image, reshape_image=reshape_image, to_categorical=to_categorical, cat_dtype=cat_dtype, augmentation=augmentation)
            
            repeat_factor_iter = 0
            while(repeat_factor_iter<repeat_factor):
                frames, decisions = self.getFileChunkArray(df=chunked_df, gray=gray, flatten=flatten, dtype=dtype, a=a, b=b, resize_shape=resize_image, reshape_image=reshape_image, to_categorical=to_categorical, cat_dtype=cat_dtype, augmentation=augmentation)    
                yield (frames, decisions)
                repeat_factor_iter+=1

            file_iter+=batch_size

def test(data_dir):
    p = PrepareDataset(data_dir)
    print("Creating Master DataFrame\n")
    df = p.createMasterDataframe()
    print("")
    p.saveFile(df=df, mode='master')
    print("Saved Master DataFrame\n")
    print("")
    print("Spliting up now ;)\n")
    p.train_test_split(validation=True)
    return p
