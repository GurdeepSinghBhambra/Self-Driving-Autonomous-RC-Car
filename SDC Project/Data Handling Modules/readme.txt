This folder has data directory named: 'data'. DO NOT CHANGE ANY FILE NAME OR FILE TYPE IN 'data' DIRECTORY

data directory contents:

data----
	|-scene1-
   		 |- videos--
		 |	    |- .avi files
		 | 
		 |- file.csv		

Note: The videos directory has video per each decision. That means for FWD command the is a video each time the command is present in the file.csv.


To extract data, use the "LoadDataset.py" library.

To process data, use the "PrepareDataset.py" library.

dependencies of LoadDataset library are:
	numpy, cv2, pandas

dependencies of PrepareDataset library are:
	sklearn, LoadDataset, pandas, numpy, cv2, keras/tensorflow

LoadDataset:
	Previous Version: 1.0.0
	Previous version: 1.1.0 (few bug fixes, new functions)
	Previous Version: 1.1.1 (few bug fixes)
	Latest Version: 1.1.2 (minor additions and changes to couple of functions)

PrepareDataset:
	Previous Version: 1.0.0
	Previous Version: 1.0.1	(minor changes)
	Previous Version: 1.0.2 (minor changes)
	Latest Version: 1.1.0 (major additions, new functions)

LoadDataset library has following Classes and functions:
	
	VideoDatasetHandler: This class handles all type of video extractions and converts it to numpy arrays.
		             The Object of this class is called by just giving the data directory path.
			     example: VideoDatasetHandler('path_of_data_directory') # here our directory name is 'data'
			
	Functions in VideoDatasetHandler as per your use:
 	
					checkDir():
						input: None
						returns: None
						Description: "checks the given directory for correct format and files, if incorrect stops the python execution"					
						Note: this function is called during object creation. No need to call it again.
  
					displayDirReport():
						input: None
						returns: None
						Description: "Displays info about the data directory"

					getNumpyVidArray(vid_file, gray=False, flatten=False, dtype=None):
						input: vid_file: path of the video file
						       gray: converts rgb image to gray image
						       flatten: converts array to 1d array using
						       dtype: data type of array to be converted to 
						returns: array of all frames in the video
						Description: "Extracts frames from 1 video only"
						Warning: If you use this then make sure to read the csv file on your own for the video's corresponding decision

			      		getAllSceneVidArray(scene_dir, gray=False, flatten=False, skip=False, dtype=None):
						input: scene_dir: path of the scene dir in the data directory
						       gray: converts rgb image to gray image
						       skip: Skip files when they fail to open, by default its False
						       flatten: converts array to 1d array using
						       dtype: data type of array to be converted to 
						returns: tuple of (array of all frames in the video, decisions for each frame) # in this order
						Description: "Extracts frames from all the videos in a particular scene directory"
						Update: This performs faster when changing dtype but the image values will not normalize. (see Examples)	
					
			      		getAllVidArray(gray=False, flatten=False, dtype=None, skip=False):
						input: gray: converts rgb image to gray image
						       skip: Skip files when they fail to open, by default its False
						       flatten: converts array to 1d array using
						       dtype: data type of array to be converted to 
						returns: tuple of (array of all frames in the video, decisions for each frame) # in this order
						Description: "Extracts frames from all the videos in the whole data directory"
						Warning: Before using the above functions make sure you have calculated for ram resorces since your model will also use ram.
						Update: This uses getAllSceneVidArray, changes apply accordingly.
						
			      (updated) showNumpyArray(nparr, tag=False, window_refresh_delay=34):
						input: nparr: numpy array having all the frames
						       tag (if given): decisions for corresponding frames
						       window_refresh_delay: minimum delay between each frame
						returns: None
						Description: "Displays frames with tags or without it"
						Update: Minor changes done

				  	cvtForOpencvImshow(nparr):
						input: nparr: numpy array which is not scaled for viewing through cv2.imshow function
						returns: Normalized array image which can be viewed in cv2.imshow
						DescriptionL: "showNumpyArray uses cv2.imshow to display video, but cv2.imshow cannot display image when image is
							      converted from uint8 to any other dtype. So, use this function to get a normalized array to view it, 
							      or even to use if you know what you are using."
						Warning: Normalizing array increase its values, that is instead of 0-255, u get have 0 - 16711680 with dtype=int32.
							 Not all dtypes are supported by cv2.imshow, so even if this function will work, you may get error or undesired image.

				  	normalize(nparr, a, b):
						input: nparr: numpy array
						       a: lower limit of interval
						       b: higher limit of interval
						returns: Normalized array
						Description: "Normalizes array values betweeen given interval"
						Warning: Normalizing array can increase value for big intervals 

				  	getAllSceneDirPaths():
						input: None
						returns: python list of all the scene directory paths in the data directory
						Description: "Returns list of all the scene directory paths in the data directory"
					
			      (updated) vidTagGen(gray=False, flatten=False, skip=False, dtype=None, get_file=False):
						input: gray: converts rgb image to gray image
						       flatten: converts array to 1d array using
						       skip: Skip files when they fail to open, by default its False
						       dtype: data type of array to be converted to
						       get_file: returns video file name from which the data is extracted
						returns: return_status, video_frames, video_decision # in this order if get_file = False
							 return_status, video_frames, video_decision, vid_file # in this order if get_file = True
						Description: "Extracts all the videos in data directory but returns it one by one only when the function is called." (see examples)	
						Extended Info:
							 return_status variable is similar to ret (in ret, frame = cv2.cap.read(), where cap = cv2.VideoCapture()). This variable is True
							 only when there is data available otherwise it is False.
						Warning: If all the videos are iterated or when an index is out of its limits, the function gets reset to first video and starts from
							 first video again from next function call. 
					
				  	setSceneDirIter(set_by_value):
						input: set_by_value: value of iterator to be set with.
						returns: None
						Description: "Sets the scene directory iterator value. This is used for setting scene dir iterator to be used with vidTagGen"
						Warning: Any positive value that exceeds the max index value (here max index = max scene dirs), resets the vidTagGen Function
				
				   	setSceneFileIter(set_by_value):
						input: set_by_value: value of iterator to be set with.
						returns: None
						Description: "Sets the scene directory file iterator value. This is used for setting scene dir file iterator to be used with vidTagGen"
						Warning: Any positive value that exceeds the max index value (here max index = max scene dir file rows), resets the vidTagGen Function
				
				  	onlyDecision(decisions):
						input: decision: numpy array having decision string of length 11 (11 is the default string length in the file.csv)	
						returns: decision numpy array having decision command only.
						Description: "Trims the original strings (eg: "b'FWD050\\n'") to contain only decision command (eg: "FWD")"
	
				  	onlySpeed(decisions):
						input: decision: numpy array having decision string of length 11 (11 is the default string length in the file.csv)	
						returns: decision numpy array having speed only.
						Description: "Trims the original strings (eg: "b'FWD050\\n'") to contain only speed (eg: "050")"

				  	onlySpeedAndDecision(decisions):
						input: decision: numpy array having decision string of length 11 (11 is the default string length in the file.csv)	
						returns: decision numpy array having decision command and speed only.
						Description: "Trims the original strings (eg: "b'FWD050\\n'") to contain only decision command and speed (eg: "FWD050")"
					
				  	str2IntDict(decisions):
						input: decision: numpy array having decision strings
						returns: python dictionary of corresponding int values for each unique decision in the decisions numpy array
						Description: "Returns dictionary having int value for each string"
					
				  	cvtStr2Int(decisions, dtype='uint8'):
						input: decision: numpy array having decision strings
						returns: numpy array of same shape (as decisions) having only positive int values with a new dtype
						Description: "Converts string numpy array having decisions to integer array having int values for corresponding strings."  		
						Warning: if no data type is passed, the new int numpy array is of dtype uint8.
					
				  	intDict2StrDict(str2int_dict):
						input: str2int_dict: python dictionary as returned by str2intDict function.
						output: python dictionary having strings for each int value. 
						Description: "Returns dictionary having decision string for each int value. (reverses dictionary from str2IntDict function)" 
				
				  	saveNumpyArrayInfo(nparr, file_path):
						input: nparr: numpy array
						       file_path: text file path in string format
						returns: None
						Description: "Saved the numpy array info into a text file"


Previous Version Examples:

here, my data directory name is "data". so use for yours accordingly


>>>from LoadDataset import VideoDatasetHandler as vdh
>>>d = vdh("data")

>>>d.displayDirReport()

>>>video_file_path = "data"+"\\"+"scene1"+"\\"+"vid1.avi" #instead of vid1.avi ucan specify which video file to extract

>>>#frames has all rgb frames from the video
>>>vid_frames = d.getNumpyVidArray(vid_file_path)

>>>#frames has converted gray frames from the videos
>>>vid_frames = d.getNumpyVidArray(vid_file_path, gray=True)

>>>#frames has flatten converted gray frames from the videos 
>>>vid_frames = d.getNumpyVidArray(vid_file_path, gray=True, flatten=True)

>>>#frames has flatten converted gray frames from the videos of dtype 'float32'
>>>vid_frames = d.getNumpyVidArray(vid_file_path, gray=True, flatten=True, dtype='float32')

>>>scene_dir_path = "data"+"\\"+"scene1"

>>>#frames has all rgb frames from all the videos in the scene dir
>>>#decisions has all the deicion for the corresponding frames from the scene dir 
>>>scene_frames, decisions = d.getAllSceneVidArray(scene_dir_path)

>>>#get all the frames and its corresponding decisions in all videos in the data directory
>>>frames, decisions = d.getAllVidArray()

>>>#shows only video
>>>d.showNumpyArray(vid_frames)

>>>#shows videos with decisions
>>>d.showNumpyArray(frames, decisions)



Latest Version Examples:

here, my data directory name is "data". so use for yours accordingly


>>>from LoadDataset import VideoDatasetHandler as vdh
>>>d = vdh("data")

>>>#get all the frames and its corresponding decisions in all videos in the data directory
>>>frames, decisions = d.getAllVidArray()

#Before Displaying using showNumpyVidArray normalize the array with cvtForOpencvImshow 
#this will greatly increase memory, use accordingly 
>>>frames = d.cvtForOpencvImshow(frames)
>>>d.showNumpyArray(frames, decisions)

#normalizing int numpy array
# here np.arrange(10) array is [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] of dtype=int32
>>> d.normalize(np.arange(10), a=100, b=200)
array([100, 111, 122, 133, 144, 155, 166, 177, 188, 199])

#normalising float numpy array
>>> d.normalize(np.arange(10).astype('float64'), a=100, b=200)
array([100.        , 111.11111111, 122.22222222, 133.33333333,
       144.44444444, 155.55555556, 166.66666667, 177.77777778,
       188.88888889, 200.        ])

#getAllSceneDirPaths
#here in my data directory only one scene dir was present.
>>> d.getAllSceneDirPaths()
['data\\scene1']

#Calling vidTagGen for first Time
>>>ret, frames, decision = d.vidTagGen()
#if ret true means there are frames and decisions
>>> ret
True
#Shape of first video
>>> frames.shape
(59, 480, 640, 3)
#decision of first video (single decision but in shape with each frame of video)
>>> decision.shape
(59,)

#using vidTagGen to iterate through the whole data dir one by one. Saves lot of ram.
>>>while(True):
>>>    ret, x, y, vid_file = d.vidTagGen(get_file=True) #now returns the video_file from which the data is extracted
>>>    if(ret == False):
>>>        continue
>>>    #do anything here, it file iterate through all the videos in the data directory while keeping ram usage very low.
>>>    #If model is trained here it will be called batch learning
>>>    #d.showNumpyArray(x, y)

#setting up scene dir iter to an out of bound index value
>>> d.setSceneDirIter(100)
#calling vidTagGen with Latest iterator values
>>> ret, frames, decision = d.vidTagGen()
#False because the directory has only 1 scene dir, so allowed index is 0
>>> ret
False
#you get nothing
>>> frames, decision
(None, None)
#again calling the function, BUT IT GOT RESET THE LAST TIME SINCE AND OUT OF BOUND POSITIVE INDEX WAS USED 
>>> ret, frames, decision = d.vidTagGen()
#It returns True because it again started from First video in first row of file.csv in first scene dir in data dir
>>> ret
True
#first frame
>>> frames.shape
(59, 480, 640, 3)
>>> decision.shape
(59,)

#set File iterator in current scene dir (currently in scene1 dir, index = 0)
#set to row with index=20 in file.csv
>>>d.setSceneFileIter(20)
#extract frames and its decision at row with index = 20 in file.csv
>>> ret, frames, decision = d.vidTagGen()
#confirm if it is present
>>> ret
True
#frame and decision is file.csv at row index=20 
>>> frames.shape
(32, 480, 640, 3)
>>> decision.shape
(32,)

#vidTagGen Starting from first video file
>>>ret, frames, decision = d.vidTagGen()
#from the original decision string (eg: "b'FWD050\\n'") convert it to having only decision
>>>d.onlyDecision(decision)
array(['FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD',
       'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD',
       'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD',
       'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD',
       'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD',
       'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD', 'FWD',
       'FWD', 'FWD', 'FWD', 'FWD', 'FWD'], dtype='<U3')

#from the original decision string (eg: "b'FWD050\\n'") convert it to having only speed
>>> d.onlySpeed(decision)
array(['050', '050', '050', '050', '050', '050', '050', '050', '050',
       '050', '050', '050', '050', '050', '050', '050', '050', '050',
       '050', '050', '050', '050', '050', '050', '050', '050', '050',
       '050', '050', '050', '050', '050', '050', '050', '050', '050',
       '050', '050', '050', '050', '050', '050', '050', '050', '050',
       '050', '050', '050', '050', '050', '050', '050', '050', '050',
       '050', '050', '050', '050', '050'], dtype='<U3')

#from the original decision string (eg: "b'FWD050\\n'") convert it to having only speed and decision
>>>d.onlySpeedAndDecision(decision)
array(['FWD050', 'FWD050', 'FWD050', 'FWD050', 'FWD050', 'FWD050',
       'FWD050', 'FWD050', 'FWD050', 'FWD050', 'FWD050', 'FWD050',
       'FWD050', 'FWD050', 'FWD050', 'FWD050', 'FWD050', 'FWD050',
       'FWD050', 'FWD050', 'FWD050', 'FWD050', 'FWD050', 'FWD050',
       'FWD050', 'FWD050', 'FWD050', 'FWD050', 'FWD050', 'FWD050',
       'FWD050', 'FWD050', 'FWD050', 'FWD050', 'FWD050', 'FWD050',
       'FWD050', 'FWD050', 'FWD050', 'FWD050', 'FWD050', 'FWD050',
       'FWD050', 'FWD050', 'FWD050', 'FWD050', 'FWD050', 'FWD050',
       'FWD050', 'FWD050', 'FWD050', 'FWD050', 'FWD050', 'FWD050',
       'FWD050', 'FWD050', 'FWD050', 'FWD050', 'FWD050'], dtype='<U6')

#checking Decision contents in first video of the first scene dir in the data dir
>>> decision
array(["b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'",
       "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'",
       "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'",
       "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'",
       "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'",
       "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'",
       "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'",
       "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'",
       "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'",
       "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'",
       "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'",
       "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'",
       "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'",
       "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'",
       "b'FWD050\\n'", "b'FWD050\\n'", "b'FWD050\\n'"], dtype='<U12')

#converting string to its corresponding int array (SEE THE DATA TYPE)
>>> d.cvtStr2Int(decision)
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=uint8)
>>> d.cvtStr2Int(decision, dtype='int16')
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=int16)

#used just for examples, extract all the videos and decisions 
>>>frames, decision = d.getAllVidArray()

#str2IntDict 
>>> d.str2IntDict(decision)
{4: "b'BKW050\\n'", 2: "b'FLF050\\n'", 3: "b'FRG050\\n'", 1: "b'FWD050\\n'", 7: "b'STP000\\n'"}

#intDict2StrDict
>>> d.intDict2StrDict(d.str2IntDict(decision))
{"b'BKW050\\n'": 4, "b'FLF050\\n'": 2, "b'FRG050\\n'": 3, "b'FWD050\\n'": 1, "b'STP000\\n'": 7}

-----------------------------------------------------------------------------------------------------------------

PrepareDataset creates a directory/folder named PreparedDataset.
so do not delete it or change the location of it or the data directory used by LoadDataset.py after the creation of master dataframe

PrepareDataset has following classes and functions:
	PrepareDataset: This class prepares and process the data for training neural network models, giving you some basic functions adapted to the current data.
			This class inherits VideoDatasetHandler class, so u can also use its functions with PrepareDataset object.
			The Object of this class is called by just giving the data directory path.
			     example: PrepareDataset('path_of_data_directory') # here our directory name is 'data' (same one which is used by VideoDatasetHandler)				

	Functions in PrepareDataset as per your use:
					openFile(mode):
						inputs: mode: specify with file to open. accepted modes: 'train', 'test', 'master', 'validation'
						returns: Pandas dataframe 
						Description: "opens a csv file and returns pandas dataframe"
				
					saveFile(df, mode):
						inputs: mode: specify the file to save to. accepted modes: 'train', 'test', 'master', 'validation'
							df: pandas dataframe (in correct format)	
							returns: True if successful else False
						Description: "saves pandas dataframe to csv file"

					getFileRows(mode, df=None):
						inputs: mode: specify the file mode. accepted modes: 'train', 'test', 'master', 'validation' or None
							df: pandas dataframe (in correct format)
						returns: integer (total rows in the file)
						Description: "gets the rows of a file"
						Note: If mode is None, then df must have a dataframe.
					
					removeDecisions(df, remove_list):
						inputs: df: pandas dataframe (in correct format)
							remove_list: python list having decisions to delete from the given dataframe
						returns: pandas dataframe (after removing the decisions or strings in df)
						Description: "removes the given list of decisions from the dataframe"
					
					removeHighestOccurringDecision(df, rows_to_delete, print_it=False):
						intputs: df: pandas dataframe (in correct format)
							 rows_to_delete: positive integer specify the total number of rows to delete
							 print_it: Boolean value indicating whether to print the label and index deleted from the dataframe
						returns: pandas dataframe (after deletion)
						Description: "Deletes some rows from the dataframe based on the criteria of highest occuring label, which ever label is highest occuring from 
							      in the dataframe it gets deleted"
					
					resize_image(image, shape, print_it=True):
						inputs: image: numpy image array
							shape: shape of the image to resize to.
							print_it: boolean value for printing warnings
						returns: resized numpy image array
						Description: "Resizes given image array"
						Note: Default frame size is (480, 640), resize to smaller shape, since increasing it also increases data and will be slow too. Also adds
						      additional data.

					createMasterDataframe(skip=False, clean_and_cvt=True, remove_cmds=['BRG', 'BLF']):
						inputs: skip: boolean value indicating whether to skip files or not.
								clean_and_cvt: boolean value whether to convert the default string to int or not.
								remove_cmds: python list specifying which commands/decisions to delete. Make sure the format is same as default strings.
						returns: pandas dataframe (Master Dataframe)
						Description: "Creates a csv file specifying frames, video_file_paths and decision/frame".
						Note: This function is very imp, without it this library has not much use. Its time intensive but you need to call it once and no need to 
						      call it in every program execution if you save the file using saveFile(df, mode='master') function.
					
					getChunkOfFile(start_index, end_index, mode, df=None):
						inputs: start_index: starting index to extract rows/chunk of dataframe/csv file
							end_index: last index to extract rows/chunk of dataframe/csv file
							mode: specify the file mode. accepted modes: 'train', 'test', 'master', 'validation' or None
							df: pandas dataframe (in correct format)
						returns: pandas dataframe (sliced dataframe for given original dataframe)
						Description: "Slices the given dataframe as per the start_index and end_index"
						Note: If mode is None, then df must have a dataframe.

			  	  (new) setRotationRange(self, lower_index, higher_index):
			  			inputs: lower_index: integer specifying lower index of brightness (range 1-100)
			  					higher_index: integer specifying higher index of brightness (range 1-100)
			  			returns: None
			  			Description: "Function to set the brightness range for transforming frames"

			          (new) setRotationRange(self, lower_index, higher_index):
			  			inputs: lower_index: integer specifying lower index of rotation
			  					higher_index: integer specifying higher index of rotation
			  			returns: None
			  			Description: "Function to set the random theta selection range for rotating frames"

		              (updated) getFrame(vid_file, frame_index, gray=False, flatten=False, dtype=None, resize_shape=None, reshape_image=None, augmentation=False):
						inputs: vid_file: path of video file to extract frame
							frame_index: index of frame to extract from video file
							gray: boolean value specifying to convert RGB image to GRAY image
							flatten: boolean value specifying to convert n-D dimentional image to 1-D image, where n is {2, 3, 4}
							dtype: changes the image/frame datatype.
							resize_image: resize tuple specifying shape of new image
							reshape_image: reshape tuple specifying the shape of new reshaped image
							augmentation: boolean value for applying the transformation for random brightness and rotation in frames.
						returns: numpy image/frame array
						Description: "Extracts frames for given frame index and video file path. Also can perform image related basic operations"
					
	                      (updated) getFileChunkArray(df, gray=False, flatten=False, dtype=None, a=None, b=None, resize_shape=None, reshape_image=None, to_categorical=False, cat_dtype=None, augmentation=False):
						inputs: df: pandas dataframe (in correct format)
							gray: boolean value specifying to convert RGB image to GRAY image
							flatten: boolean value specifying to convert n-D dimentional image to 1-D image, where n is {2, 3, 4}
							dtype: changes the image/frame datatype.
							a: lower limit for normalizing each frame/image.
							b: higher limit for normalizing each frame/image.
							resize_image: resize tuple specifying shape of new image
							reshape_image: reshape tuple specifying the shape of new reshaped image
							to_categorical: changes the int decisions to one-hot encoding
							cat_dtype: changes datatype of the one-hot encoded numpy array (default dtype is uint8)
							augmentation: boolean value for applying the transformation for random brightness and rotation in frames.
						returns: video_frames, video_decision #in this order
						Description: "Extracts the video_frames and their decisions from the given dataframe"

					train_test_split(df=None, ret=False, validation=False, test_size=0.15, validation_size=0.15, random_state=None, shuffle=False, stratify=False):
						inputs: df: pandas dataframe (in correct format)
							ret: boolean value specifying whether to return the split dataset arrays or not.
							validation: boolean value specifying whether to split the dataset into 2 sets (train, test) or 3 sets(train, set, validation)
							test_size: float value specifying the size of test dataset wrt master dataset (master dataset is master dataframe)
							validation_size: float value specifying the size of validation dataset wrt master dataset (master dataset is master dataframe)
							random_state: same as the random_state in sklearn train_test_split
							shuffle: boolean value specifying whether to shuffle or not (same as the shuffle in sklearn train_test_split)
							stratify: boolean value specifying whether to stratify or not (equal to startify in sklearn train_test_split but takes only boolean values)
						returns: if ret true: returns train_x, test_x, train_y, test_y or train_x, test_x, val_x, train_y, test_y, val_y (based on validation)
							 else: None
						Description: "splits the dataset into train, test, validation sub sets and also saves them in PreparedDataset directory"
						Note: Our current dataset has 3 'BRG' values and 1 'BLF' values which is by default cleaned during createMasterDataframe function.
						      if you change the default remove_list of createMasterDataframe and do not delete these decisions, while using stratify you will get error regarding 
						      unsufficient amount of classes present to split the dataset.
						Warning: Using stratify, shuffle is forced to be True even if you specify shuffle as False.

					getBatchSizes(files=['train', 'test', 'validation', 'master'], df=None):
						inputs: files: python list specifying the file modes
							df: pandas dataframe (in correct format)
						returns: python dictionary having file type and its possible batch sizes
						Description: "Gives file type and its possible batch sizes without deleting any row in the file"
						
					shapeAccordingToBatchSize(batch_size, mode, save=False, df=None):
						inputs: batch_size: batch_size for the dataset
							mode: specify the file mode. accepted modes: 'train', 'test', 'master', 'validation' or None
							save: boolean value specifying whether to save the new trimmed file or not. Does not work for mode = None
							df: pandas dataframe (in correct format)
						returns: trimmed pandas dataframe
						Description: "Trims the Dataframe according to batch_size"

					categorical2Int(cat_array):
						inputs: cat_array: one-hot encoded numpy array
						returns: numpy integer 1-D array
						Description: "Converts the given one-hot encoded array to 1-D integer array"

			          (new) stepsPerEpoch(self, mode, batch_size, repeat_factor=1, df=None):
			  			inputs: mode: specify the file mode. accepted modes: 'train', 'test', 'master', 'validation' or None
			  					batch_size: batch_size for the dataset
		  						repeat_factor: integer specifying to repeat a single batch given no of times
		  				returns: float number
		  				Description: "Gives the steps_per_epoch value for the given mode/dataframe"
		  				Note: When repeat_factor of the generator is changed make sure you specify here to get the correct number. 

		              (updated) datasetGenerator(self, batch_size, mode, df=None, gray=False, flatten=False, dtype=None, a=None, b=None, resize_image=None, reshape_image=None, to_categorical=False, cat_dtype=None, augmentation=False, repeat_factor=1):
						inputs: batch_size: batch_size for the dataset
							mode: specify the file mode. accepted modes: 'train', 'test', 'master', 'validation' or None
							df: pandas dataframe (in correct format)
							gray: boolean value specifying to convert RGB image to GRAY image
							flatten: boolean value specifying to convert n-D dimentional image to 1-D image, where n is {2, 3, 4}
							dtype: changes the image/frame datatype.
							a: lower limit for normalizing each frame/image.
							b: higher limit for normalizing each frame/image.
							resize_image: resize tuple specifying shape of new image
							reshape_image: reshape tuple specifying the shape of new reshaped image
							to_categorical: changes the int decisions to one-hot encoding
							cat_dtype: changes datatype of the one-hot encoded numpy array (default dtype is uint8)
							augmentation: boolean value for applying the transformation for random brightness and rotation in frames.
							repeat_factor: integer specifying to repeat a single batch given no of times
						returns: python generator object
						Description: "This is a infinite loop Python generator for iterating over the dataset in batches. Use this function for keras model.fit_generator"
						Note: Make sure you pay attention to any error, warnings or statements produced by this function


Examples:
	
>>>from PrepareDataset import PrepareDataset as ppd

>>>#make class object 
>>>p = ppd("data") # "data" is my directory 

>>>#create master dataframe
>>>df = p.createMasterDataframe()

>>>#save the master dataframe so that you dont need to call it in every program execution
>>>p.saveFile(df=df, mode='master')
					
>>>#get chunk of a file
>>>#gets 2 rows from index=700 in master dataframe
>>>df = p.getChunkOfFile(start_index=700, end_index=702, mode='master', df=None) 

>>>#opens file
>>>df = p.openFile(mode='master')
>>>#same as above but by passing dataframe. Note the mode is None and df is given the dataframe, in this manner pass any dataframe to functions having such feature 
>>>df = p.getChunkOfFile(start_index=700, end_index=702, mode=None, df=df) 

>>>#get numpy arrays for the sliced dataframe above
>>>frames, tags = p.getFileChunkArray(df)
>>>frames.shape, tags.shape
(2, 480, 640, 3), (2, )

>>>#creates 3 files (since validation is True else it will be 2 files) from the dataset
>>>p.train_test_split(ret=False, validation=True, test_size=0.15, validation_size=0.25, stratify=True)

>>>#saves the 3 sets also returns it
>>>train_x, test_x, val_x, train_y, test_y, val_v = p.train_test_split(ret=True, validation=True, test_size=0.15, validation_size=0.25, stratify=True)

>>>#saves 2 sets (train, test) and also returns them
>>>train_x, test_x, train_y, test_y = p.train_test_split(ret=True, validation=False, test_size=0.15, shuffle=True)
>>>#saves 2 sets (train, test) but doesn't returns them
>>>p.train_test_split(test_size=0.20)

>>>#get batch sizes for all files
>>>p.getBatchSizes()
{'train': [1, 3, 5, 15, 2521, 7563, 12605, 37815], 'test': [1, 2, 4, 8, 1013, 2026, 4052, 8104], 'validation': [1, 2, 4, 8, 1013, 2026, 4052, 8104], 'master': [1, 89, 607, 54023]}

>>>#get file size/rows as shown below
>>>print("\nBefore deletion, master file size:", p.getFileRows(mode='master'))
Before deletion, master file size: 54023

>>>#trim dataset according to your desired batch_size
>>>trimmed_df = p.shapeAccordingToBatchSize(batch_size=32, mode='master')
Total rows deleted to match the batch size = 7

>>>print("\nAfter deletion, master file size:", p.getFileRows(mode=None, df=trimmed_df))
After deletion, master file size: 54016


# Use the generator in for model.fit_generator but you can also use it like shown below
>>>for frames, decisions in p.datasetGenerator(batch_size=32, mode='train', gray=True, resize_image=(480, 200), dtype=np.float32, a=0, b=1, reshape_image=(480, 200, 1), to_categorical=True, cat_dtype=np.int64, augmentation=True):
>>>     print("received from generator")
>>>     print(frames.shape, frames.dtype)
>>>     print(decisions.shape, decisions.dtype)
>>>     print(frames[0])
>>>     print(decisions[0])
>>>     break
Total rows deleted to match the batch size = 23

received from generator
(32, 480, 200, 1) float32
(32, 8) int64
[[[0.14509805]
  [0.15294118]
  [0.16470589]
  ...
  [0.91372555]
  [0.9058824 ]
  [0.90196085]]

 [[0.8941177 ]
  [0.882353  ]
  [0.8705883 ]
  ...
  [0.7058824 ]
  [0.7058824 ]
  [0.7058824 ]]

 [[0.7058824 ]
  [0.7058824 ]
  [0.7019608 ]
  ...
  [0.5921569 ]
  [0.5921569 ]
  [0.5921569 ]]

 ...

 [[0.4431373 ]
  [0.4431373 ]
  [0.4431373 ]
  ...
  [0.8078432 ]
  [0.78823537]
  [0.7843138 ]]

 [[0.7803922 ]
  [0.7803922 ]
  [0.77647066]
  ...
  [0.43529415]
  [0.43529415]
  [0.43529415]]

 [[0.43529415]
  [0.43529415]
  [0.43137258]
  ...
  [0.40000004]
  [0.40000004]
  [0.40000004]]]
[0 1 0 0 0 0 0 0]


#get the steps_per_epoch like shown below:
>>> print(p.stepsPerEpoch(mode='train', batch_size=10, repeat_factor=2, df=None))
5322.0

>>>  print(p.stepsPerEpoch(mode='test', batch_size=10, repeat_factor=2, df=None))
818.0

Check the combine_collage_datasets.py file. Open it, Read it, Change it, Use it.

I haven't included some small functions in the examples just because they are pretty self-explanatory and easy to use.
Still if you have any doubts ask right away.

If you directly want to get started refer to example below. Use the doMeDaddy Function as shown below

eg:

from PrepareDataset import PrepareDatset as ppd
from PrepareDataset import doMeDaddy

# open the library and read the 'use_like_this' function (ek hi comment hai)

#doMeDaddy function does all the work for you
# it creates the datframe, saves it
# then it splits the dataset into 3 sets(train, test, split) with test_size=0.15, validation_size=0.15, shuffle=False, stratify=False
# returns a PrepareDataset object 

p = doMeDaddy(path_of_data_directory) # path_of_data_directory refers to data directory used by LoadDataset

# declare the generator, this is a infinite loop generator made for keras.sequential.fit_generator
train_gen = p.datasetGenerator(batch_size=32, mode='train', gray=True, to_categorical=True, reshape_image=(480, 640, 1))
test_gen = p.datasetGenerator(batch_size=32, mode='test', gray=True, to_categorical=True, reshape_image=(480, 640, 1))
validation_gen = p.datasetGenerator(batch_size=32, mode='validation', gray=True, to_categorical=True, reshape_image=(480, 640, 1))


