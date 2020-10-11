__author__ = "Gurdeep"


import os
import numpy as np
import cv2
import pandas as pd


class VideoDatasetHandler:
    def __init__(self, master_directory):
        self.master_directory = master_directory
        self.parent_of_master_directory = ''
        self.scene_dir_iterator = 0
        self.scene_file_iterator = 0
        self.checkDir()
        self.scene_dir_list = sorted([self.master_directory+"\\"+scene_dir for scene_dir in os.listdir(self.master_directory)])

    def checkDir(self):
        if(os.path.isdir(self.master_directory) == False):
            print("ERROR in checkDir Function : {} Directory Not Found".format(self.master_directory))
            exit(0)
        master_dir_list = os.listdir(self.master_directory)
        if("\\" in self.master_directory):  
            self.parent_of_master_directory = "\\".join(self.master_directory.split('\\')[0:-1])
        elif("/"in self.master_directory):  
            self.parent_of_master_directory = "\\".join(self.master_directory.split('/')[0:-1])
        if(len(master_dir_list) == 0):
            print("ERROR in checkDir Function: \"{}\" Master Directory Empty".format(self.master_directory))
            exit(0)
        for d in master_dir_list:
            if('.' in d):
                print("ERROR in checkDir Function: {} Master Directory Format Incorrect, Should Only Contain Scene Directories".format(self.master_directory))
                exit(0)
            d = self.master_directory+"\\"+d
            curr_dir_list = os.listdir(d)
            if(len(curr_dir_list) == 0):
                print("ERROR in checkDir Function: \"{}\" Scene Directory Empty".format(d))
                exit(0)
            if(len(curr_dir_list) > 2):
                print("ERROR in checkDir Function: \"{}\" Scene Directory Format Incorrect, Contains Unknown Files".format(d))
                exit(0)
            if("file.csv" not in curr_dir_list):
                print("ERROR in checkDir Function: \"{}\" Scene Directory Does Not Contain \"file.csv\"".format(d))
                exit(0)
            if("videos" not in curr_dir_list):
                print("ERROR in checkDir Function: \"{}\" Scene Directory Does Not Contain \"videos\" directory".format(d))
                exit(0)
            vid_dir = d+"\\"+"videos"
            if(len(os.listdir(vid_dir)) == 0):
                print("ERROR in checkDir Function: \"{}\" Video Directory Empty".format(d+"\\"+"videos"))
                exit(0)
            flag = True
            for vid in os.listdir(vid_dir):
                if(vid.split('.')[-1] != "avi"):
                    flag = False
                    break
            if(flag == False):
                print("ERROR in checkDir Function: {} Video Directory Should Only Contain \".avi\" files".format(vid_dir))
                exit(0)

    def displayDirReport(self):
        total_scene_dirs = len(os.listdir(self.master_directory))
        print("Master Directory Summary:")
        print("\tMaster Directory:", self.master_directory)
        print("\tNo of Scene Directories:", total_scene_dirs)
        print("\tScene Directory Summary:")
        for d_no, d in enumerate(os.listdir(self.master_directory), 1):
            d = self.master_directory+"\\"+d
            d_lst = os.listdir(d)
            print("\t\t****************************************************")
            print("\t\tScene Directory {}/{}".format(d_no, total_scene_dirs))
            print("\t\tScene Directory Path:", d)
            print("\t\tScene Directory Contents:", d_lst)
            print("\t\tNo of Videos:", len(os.listdir(d+"\\"+"videos")))
            print("\t\t****************************************************\n")
        print("\n")

    @staticmethod
    def getNumpyVidArray(vid_file, gray=False, flatten=False, dtype=None):
        cap = cv2.VideoCapture(vid_file)
        vid_frames = None
        i=0
        while(True):
            ret, frame = cap.read()
            if(not ret):
                break
            if(gray == True):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if(flatten == True):
                frame = frame.flatten()
            if(i==0):
                shape = list()
                shape.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                shape.extend(list(frame.shape))
                if(dtype == None):
                    vid_frames = np.empty(shape, dtype=frame.dtype)
                else:
                    vid_frames = np.empty(shape, dtype=dtype)
            if(dtype == None):
                vid_frames[i] = frame
            else:
                vid_frames[i] = frame.astype(dtype)
            i+=1
        cap.release()
        return vid_frames

    @staticmethod
    def getSceneVidInfo(vid_file, gray=False, flatten=False):
        cap = cv2.VideoCapture(vid_file)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret = False
        while(not ret):
            ret, frame = cap.read()
        cap.release()
        if(gray == True and flatten == True):
            return (frames, frame.shape[0]*frame.shape[1], frame.dtype)
        elif(gray == True and flatten == False):
            return (frames, (frame.shape[:-1]), frame.dtype)
        else:
            return frames, frame.shape, frame.dtype

    def getAllSceneVidArray(self, scene_dir, gray=False, flatten=False, skip=False, dtype=None):
        try:
            file = pd.read_csv(scene_dir+'\\'+'file.csv')
        except Exception as exx:
            print("ERROR in getAllSceneVidArray Function:", exx)
            exit(0)

        print("\tInitializing for capturing all videos in {} scene dir ...".format(scene_dir))
        total_frames, frame_shape, frame_dtype = 0, None, None
        for vid_file in file.video_file:
            if(self.parent_of_master_directory != ""):
                vid_file = self.parent_of_master_directory+"\\"+vid_file
            if(os.path.isfile(vid_file) == False and skip == True):    
                print("\t{} file not present, skipped".format(vid_file))
                continue
            elif(os.path.isfile(vid_file) == False and skip == False):
                print("ERROR in getAllSceneVidArray Function: {} file not present".format(vid_file))
                return None, None
            frames, frame_shape, frame_dtype = self.getSceneVidInfo(vid_file, gray=gray, flatten=flatten)
            total_frames += frames
        print("\tInitialization Complete.\n\tSummary:")
        print("\t\tVideos dir path:", scene_dir+"\\"+"videos")
        print("\t\tNo of videos:", file.video_file.count())
        print("\t\tNo of Frames in video dir:", total_frames)
        print("\t\tFrame Shape:", frame_shape)
        
        shape = list()
        shape.append(int(total_frames))
        if(flatten == True):
            shape.append(frame_shape)
        else:
            shape.extend(list(frame_shape))
        dir_decisions = np.empty((total_frames, ), dtype='<U12')
        if(dtype == None):
            print("\tFrame dtype:", frame_dtype)
            dir_frames = np.empty(shape, dtype=frame_dtype)
        else:
            print("\tFrame dtype:", dtype)
            dir_frames = np.empty(shape, dtype=dtype)
        i=0
        print("\n\tStacking All Videos in", scene_dir+"\\"+"videos")
        for vid_file, decision in zip(file.video_file, file.decision):
            if(self.parent_of_master_directory != ""):
                vid_file = self.parent_of_master_directory+"\\"+vid_file
            if(os.path.isfile(vid_file) == False and skip == True):    
                print("{} file not present, skipped".format(vid_file))
                continue
            elif(os.path.isfile(vid_file) == False and skip == False):
                print("ERROR in getAllSceneVidArray Function: {} file not present".format(vid_file))
                return None, None
            vid_frames = self.getNumpyVidArray(vid_file, gray=gray, flatten=flatten, dtype=dtype)
            no_of_vid_frames = vid_frames.shape[0]
            dir_frames[i:i+no_of_vid_frames] = vid_frames[:]
            dir_decisions[i:i+no_of_vid_frames] = decision
            i+=no_of_vid_frames
        print("\tStacking Scene Directory Complete.\n")
        return dir_frames, dir_decisions  

    def getAllVidInfo(self, gray=False, flatten=False, dtype=None, skip=False):
        print("Initializing for capturing all videos in \"{}\" master dir ...".format(self.master_directory))
        total_frames, total_videos, frame_shape, frame_dtype = 0, 0, None, None
        for scene_dir in os.listdir(self.master_directory):
            try:
                file = pd.read_csv(self.master_directory+"\\"+scene_dir+'\\'+'file.csv')
            except Exception as exx:
                if(skip == True):
                    print("Warning: {} file skipped".format(self.master_directory+"\\"+scene_dir+'\\'+'file.csv'))
                    continue
                else:
                    print("ERROR in getAllVidInfo Function:", exx)
                    exit(0)
            
            for vid_file in file.video_file:
                if(self.parent_of_master_directory != ""):
                    vid_file = self.parent_of_master_directory+"\\"+vid_file
                if(os.path.isfile(vid_file) == False and skip == True):    
                    print("{} file not present, skipped".format(vid_file))
                    continue
                elif(os.path.isfile(vid_file) == False and skip == False):
                    print("ERROR in getAllVidInfo Function: {} file not present".format(vid_file))
                    return None, None
                frames, frame_shape, frame_dtype = self.getSceneVidInfo(vid_file, gray=gray, flatten=flatten)
                total_frames += frames
            total_videos += file.video_file.count()
        print("Initialization Complete.\nSummary:")
        print("\tScene dirs:", os.listdir(self.master_directory))
        print("\tNo of videos:", total_videos)
        print("\tNo of Frames in all scene dirs:", total_frames)
        print("\tFrame Shape:", frame_shape)
        if(dtype == None):
            print("\tFrame dtype:", frame_dtype)
            return total_frames, frame_shape, frame_dtype
        else:
            print("\tFrame dtype:", dtype)
            return total_frames, frame_shape, dtype

    def getAllVidArray(self, gray=False, flatten=False, dtype=None, skip=False):
        total_frames, frame_shape, frame_dtype = self.getAllVidInfo(gray=gray, flatten=flatten, dtype=dtype, skip=skip)
        shape = list()
        shape.append(int(total_frames))
        if(flatten == True):
            shape.append(frame_shape)
        else:
            shape.extend(list(frame_shape))
        dir_decisions = np.empty((total_frames, ), dtype='<U12')
        dir_frames = np.empty(shape, dtype=frame_dtype)
        i=0
        print("Stacking all videos in {} master directory\n".format(self.master_directory))
        for dir_no, scene_dir in enumerate(self.scene_dir_list, 1):
            print("\t***************************************************")
            print("\tScene Dir: {}/{}".format(dir_no, len(self.scene_dir_list)))
            print("\tCurrent Scene Dir:", scene_dir)
            frames, decisions = self.getAllSceneVidArray(scene_dir, gray=gray, flatten=flatten, dtype=dtype, skip=skip)
            no_of_frames = frames.shape[0]
            dir_frames[i:i+no_of_frames] = frames
            dir_decisions[i:i+no_of_frames] = decisions
            i+=no_of_frames
            print("\t***************************************************\n")
        print("Stacking Master Directory Complete\n\n")
        return dir_frames, dir_decisions

    @staticmethod
    def cvtForOpencvImshow(nparr):
        try:    
            if('int' in str(nparr.dtype)):
                if(nparr.dtype == np.int8):
                    raise Exception("Cannot Convert Image, Array Data Type is \'int8\'")
                if(nparr.dtype == np.uint8):
                    return nparr  
                return nparr * 256
            elif('float' in str(nparr.dtype)):
                return nparr / 256
            else:
                raise Exception('Numpy Array type is neither float or integer')
        except Exception as exx:
            print("ERROR in cvtForOpencvImshow Function:", exx)
            exit(0)

    @staticmethod
    def normalize(nparr, a, b):
        try:
            if('int' in str(nparr.dtype)):
                return a + (nparr * ((b-a)//nparr.max()))
            elif('float' in str(nparr.dtype)):
                return a + (nparr * ((b-a)/nparr.max()))
            else:
                raise Exception("Invalid Numpy Array Data-type, supported dtypes are \'float\' and \'int\'")
        except Exception as exx:
            print("ERROR in normalize Function:", exx)
            exit(0)

    @staticmethod
    def showNumpyArray(nparr, tag=False, window_refresh_delay=34):
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        font_color = (0, 0, 255) # red Color, format=BGR
        font_color_for_closing = (88, 153, 255)
        #window_refresh_delay = 34 # in ms
        if(type(tag) == type(False) and tag == False):
            for frame in nparr:
                cv2.putText(frame, "Press q to exit", (0, 25), font, 0.52, font_color_for_closing, thickness, cv2.LINE_AA)
                cv2.imshow('video', frame)
                if(cv2.waitKey(window_refresh_delay) == ord('q')):
                    break
        else:
            for frame, tag in zip(nparr, tag):
                cv2.putText(frame, "Press q to exit", (0, 25), font, 0.52, font_color_for_closing, thickness, cv2.LINE_AA)
                cv2.putText(frame, str(tag), (0, 61), font, 0.61, font_color, thickness, cv2.LINE_AA)
                cv2.imshow('video', frame)
                if(cv2.waitKey(window_refresh_delay) == ord('q')):
                    break
        cv2.destroyAllWindows()

    def getAllSceneDirPaths(self):
        return self.scene_dir_list.copy()
    
    def setSceneDirIter(self, set_by_value):
        if(set_by_value < 0):
            print("ERROR in setSceneDirIter: Index Value Cannot be Less Than Zero")
            exit(0)
        self.scene_dir_iterator = set_by_value

    def setSceneFileIter(self, set_by_value):
        if(set_by_value < 0):
            print("ERROR in setSceneFileIter: Index Value Cannot be Less Than Zero")
            exit(0)
        self.scene_file_iterator = set_by_value

    def vidTagGen(self, gray=False, flatten=False, skip=False, dtype=None, get_file=False):
        if(self.scene_dir_iterator >= int(len(self.scene_dir_list))):
            self.scene_dir_iterator=0
            self.scene_file_iterator=0
            if(get_file == True):
                return False, None, None, None
            return False, None, None
        scene_dir = self.scene_dir_list[self.scene_dir_iterator]
        try:
            file = pd.read_csv(scene_dir+'\\'+'file.csv')
        except Exception as exx:
            print("ERROR:", exx)
            exit(0)

        if(self.scene_file_iterator >= file.shape[0]):
            self.scene_dir_iterator=0
            self.scene_file_iterator=0
            if(get_file == True):
                return False, None, None, None
            return False, None, None

        vid_file, decision = file.video_file[self.scene_file_iterator], file.decision[self.scene_file_iterator]
        if(self.parent_of_master_directory != ""):
            vid_file = self.parent_of_master_directory+"\\"+vid_file

        if(os.path.isfile(vid_file) == False and skip == True):    
            print("{} file not present, skipped".format(vid_file))
            if(get_file == True):
                return True, None, None, None
            return True, None, None
        elif(os.path.isfile(vid_file) == False and skip == False):
            print("ERROR in vidTagGen Function: {} file not present".format(vid_file))
            self.scene_dir_iterator = 0
            self.scene_file_iterator = 0
            if(get_file == True):
                return False, None, None, None
            return False, None, None
        
        vid_frames = self.getNumpyVidArray(vid_file, gray=gray, flatten=flatten, dtype=dtype)
        vid_decision = np.empty((vid_frames.shape[0], ), dtype='<U12')
        vid_decision[:] = decision

        if(self.scene_file_iterator+1 >= file.shape[0]):
            self.scene_dir_iterator+=1
            self.scene_file_iterator=0
        else:
            self.scene_file_iterator+=1
        
        if(get_file == True):
            return True, vid_frames, vid_decision, vid_file
        else:
            return True, vid_frames, vid_decision

    @staticmethod
    def onlyDecision(decisions):
        try:
            if(int(len(decisions[0])) != 11): 
                raise Exception("The Array is Already Modified") 
            decisions = np.vectorize(lambda x: x[2:-6])(decisions)
        except Exception as exx:
            print("ERROR in onlyDecision Function:", exx)
            exit(0)
        return decisions

    @staticmethod
    def onlySpeed(decisions):
        try:
            if(int(len(decisions[0])) != 11): 
                raise Exception("The Array is Already Modified") 
            decisions = np.vectorize(lambda x: x[5:-3])(decisions)
        except Exception as exx:
            print("ERROR in onlySpeed Function:", exx)
            exit(0)
        return decisions

    @staticmethod
    def onlySpeedAndDecision(decisions):
        try:
            if(int(len(decisions[0])) != 11): 
                raise Exception("The Array is Already Modified") 
            decisions = np.vectorize(lambda x: x[2:-3])(decisions)
        except Exception as exx:
            print("ERROR in onlySpeedAndDecision Function:", exx)
            exit(0)
        return decisions

    @staticmethod
    def str2int_values(decision):
        if('FWD' in decision):
            return 1
        elif('FLF' in decision):
            return 2
        elif('FRG' in decision):
            return 3
        elif('BKW' in decision):
            return 4
        elif('BLF' in decision):
            return 5
        elif('BRG' in decision):
            return 6
        elif('STP' in decision):
            return 7
        else:
            print("ERROR while creating dictionary in str2IntDict/cvtStr2Int: Invalid Array Value Present")
            exit(0) 

    def str2IntDict(self, decisions):
        return {self.str2int_values(x):x for x in np.unique(decisions)}
        
    def cvtStr2Int(self, decisions, dtype=np.uint8):
        return np.vectorize(self.str2int_values)(decisions).astype(dtype)

    @staticmethod
    def intDict2StrDict(str2int_dict):
        int2str_dict = dict()
        try:
            for key, value in str2int_dict.items():
                int2str_dict[value] = key
        except Exception as exx:
            print("ERROR in intDict2StrDict:", exx)
        return int2str_dict

    @staticmethod
    def saveNumpyArrayInfo(nparr, file_path):
        with open(file_path, 'w') as f:
            print("type: " + str(type(nparr)), file=f)
            print("shape: " + str(nparr.shape), file=f)
            print("strides: " + str(nparr.strides), file=f)
            print("itemsize: " + str(nparr.itemsize), file=f)
            print("dtype: " + str(nparr.dtype), file=f)
