'''
Code extracts frames from video at a rate of 25fps and 

'''
import multiprocessing as mp
from multiprocessing import Process, Pool
import threading as td
import sys, pdb, cv2, subprocess, struct
import time, os

import numpy as np
from tqdm import tqdm
TAG_FLOAT = 202021.25
MIN_WIDTH = 1
MAX_WIDTH = 99999
MIN_HEIGHT = 1
MAX_HEIGHT = 99999

def readFlowFile(filename):
    """
    flow_utils.readFlowFile(<FILENAME>) reads a flow file <FILENAME> into a 2-band np.array.

    if <FILENAME> does not exist, an IOError is raised.
    if <FILENAME> does not finish by '.flo' or the tag, the width, the height or the file's size is illegal, an Expcetion is raised.

    ---- PARAMETERS ----
        filename: string containg the name of the file to read a flow

    ---- OUTPUTS ----
        a np.array of dimension (height x width x 2) containing the flow of type 'float32'
    """
        
    # check filename
    if not filename.endswith(".flo"):
        raise Exception("readFlowFile({:s}): filename must finish with '.flo'".format(filename))
    
    # open the file and read it
    with open(filename,'rb') as f:
        # check tag
        tag = struct.unpack('f',f.read(4))[0]
        if tag != TAG_FLOAT:
            raise Exception("flow_utils.readFlowFile({:s}): wrong tag".format(filename))
        # read dimension
        w,h = struct.unpack('ii',f.read(8))
        if w < MIN_WIDTH or w > MAX_WIDTH:
            raise Exception("flow_utils.readFlowFile({:s}: illegal width {:d}".format(filename,w))
        if h < MIN_HEIGHT or h > MAX_HEIGHT:
            raise Exception("flow_utils.readFlowFile({:s}: illegal height {:d}".format(filename,h))
        flow = np.fromfile(f,'float32')
        if not flow.shape == (h*w*2,):
            raise Exception("flow_utils.readFlowFile({:s}: illegal size of the file".format(filename))
        flow.shape = (h,w,2)
        return flow       

def extract_flows(vid_dir, frame_dir, start, end, gpu, redo=False):
  
  print('start = ', start, ' end = ', end)
  class_list = sorted(os.listdir(vid_dir))[start:end]
  print("Classes =", class_list)
  for ic,cls in enumerate(class_list):
    vlist = sorted(os.listdir(vid_dir + cls))
    print("")
    print(ic+1, len(class_list), cls, len(vlist))
    print("")
    for v in tqdm(vlist):
        outdir = os.path.join(frame_dir, cls, v[:-4])
        if os.path.isfile( os.path.join(outdir, 'done') ) and not redo: continue
        try:  
            os.system('mkdir -p "%s"'%(outdir))
            # check if horizontal or vertical scaling factor
            o = subprocess.check_output('ffprobe -v error -show_entries stream=width,height -of default=noprint_wrappers=1 "%s"'%(os.path.join(vid_dir, cls, v)), shell=True).decode('utf-8')
            lines = o.splitlines()
            width = int(lines[0].split('=')[1])
            height = int(lines[1].split('=')[1])
            resize_str = '-1:256' if width>height else '256:-1'

            # extract frames
            os.system('ffmpeg -i "%s" -r 25 -q:v 2 -vf "scale=%s" "%s"  > /dev/null 2>&1'%( os.path.join(vid_dir, cls, v), resize_str, os.path.join(outdir, '%05d.jpg')))
            nframes = len([ fname for fname in os.listdir(outdir) if fname.endswith('.jpg') and len(fname)==9])
            if nframes==0: raise Exception # pdb.set_trace()

            # extract flows 
            os.system('export CUDA_VISIBLE_DEVICES=%d; ./utils1/tvl1_videoframes "%s" %d "%s" '%(gpu, os.path.join(outdir, '%05d.jpg'), nframes, os.path.join(outdir,'%05d_flow256crop.flo') ))
            nflows = len([ fname for fname in os.listdir(outdir) if fname.endswith('.flo')])
            if nflows!=nframes-1: raise Exception 

            # read flow and save jpg
            for i in range(1, nframes):
                flow = readFlowFile(os.path.join(outdir,'%05d_flow256crop.flo'%(i)))
                tflow = np.minimum(20, np.maximum(-20, flow))
                iflow = ( (tflow+20)/40.0*255.0 ).astype(np.uint8)
                cv2.imwrite(os.path.join(outdir, 'TVL1jpg_x_%05d.jpg'%(i)), iflow[:,:,0])
                cv2.imwrite(os.path.join(outdir, 'TVL1jpg_y_%05d.jpg'%(i)), iflow[:,:,1])
            # mark as done
            for i in range(1,nframes):
                os.system('rm "%s"'%(os.path.join(outdir,'%05d_flow256crop.flo'%(i))))
            os.system('touch "%s"'%(os.path.join(outdir, 'done') ))
        except:
            print("ERROR", cls, v)
def Bar(arg):  # 父进程去执行，而不是子进程调用
    print('-->exec done:', arg)
    print('in process[Bar]', os.getpid())

if __name__ == '__main__':
  vid_dir   = sys.argv[1]
  frame_dir = sys.argv[2]
  start     = int(sys.argv[3])
  end       = int(sys.argv[4])
  gpu       = int(sys.argv[5])
  # extract_flows(vid_dir, frame_dir, start, end, gpu, redo=True)


  thread_num=9
  clip=(end-start)//thread_num
  list_start=[start+i*clip for i in range(thread_num+1)]
  list_end=[start+(i+1)*clip for i in range(thread_num+1)]
  list_end[thread_num]=end
  pool = Pool(thread_num)  # 允许进程池里同时放入5个进程 其他多余的进程处于挂起状态
  # for i in range(len(list_start)):
  #     print(str(list_start[i] )+ '----' + str(list_end[i]))
  try:
      for i in range(len(list_start)):
          # t1 = td.Thread(target=extract_flows, args=(vid_dir, frame_dir,list_start[i],list_end[i],gpu,True))
          pool.apply_async(func=extract_flows, args=(vid_dir, frame_dir,list_start[i],list_end[i],gpu,True), callback=Bar)
          print("Tread "+str(i)+" is running...")
      print('end：', os.getpid())
      pool.close()  # close() 必须在join()前被调用
      pool.join()  # 进程池中进程执行完毕后再关闭，如果注释，那么程序直接关闭。
  except:
      print("Error: 无法启动线程")
  # while 1:
  #     time.sleep(5)
  #     pass
  # extract_flows(vid_dir, frame_dir, start, end, gpu, redo=True)