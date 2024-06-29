from genericpath import isfile
import imp
import time
import jittor as jt
from tqdm import tqdm
import numpy as np
import jdet
import pickle
import datetime
from jdet.config import get_cfg,save_cfg
from jdet.utils.visualization import visualize_results
from jdet.utils.registry import build_from_cfg,MODELS,SCHEDULERS,DATASETS,HOOKS,OPTIMS
from jdet.config import get_classes_by_name
from jdet.utils.general import build_file, current_time, sync,check_file,check_interval,parse_losses,search_ckpt
from jdet.data.devkits.data_merge import data_merge_result
import os
import shutil
from tqdm import tqdm
from jittor_utils import auto_diff
import copy
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
from pathlib import Path
import cv2
import glob
flag = False
from jdet.data.yolo import letterbox
class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path))  # os-agnostic
        p = os.path.abspath(p)  # absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            h0, w0 = img0.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
                img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        

        return path, img, img0, self.cap

def write():
    global flag
    if not flag:
        fw = open("/data2/zhangyong/workspace/project/yolox/JDet-master/work_dirs/yolov5s_coco_12epoch_ema/test/images.txt","w")
        flag = True
    else:
        fw = open("/data2/zhangyong/workspace/project/yolox/JDet-master/work_dirs/yolov5s_coco_12epoch_ema/test/images.txt","a+")
    return fw
class Runner:
    def __init__(self):
        cfg = get_cfg()
        self.cfg = cfg
        self.flip_test = [] if cfg.flip_test is None else cfg.flip_test
        self.work_dir = cfg.work_dir

        self.max_epoch = cfg.max_epoch 
        self.max_iter = cfg.max_iter
        assert (self.max_iter is None)^(self.max_epoch is None),"You must set max_iter or max_epoch"

        self.checkpoint_interval = cfg.checkpoint_interval
        self.eval_interval = cfg.eval_interval
        self.log_interval = cfg.log_interval
        self.resume_path = cfg.resume_path
    
        self.model = build_from_cfg(cfg.model,MODELS)
        if (cfg.parameter_groups_generator):
            params = build_from_cfg(cfg.parameter_groups_generator,MODELS,named_params=self.model.named_parameters(), model=self.model)
        else:
            params = self.model.parameters()
        self.optimizer = build_from_cfg(cfg.optimizer,OPTIMS,params=params)
        self.scheduler = build_from_cfg(cfg.scheduler,SCHEDULERS,optimizer=self.optimizer)
        self.train_dataset = build_from_cfg(cfg.dataset.train,DATASETS,drop_last=jt.in_mpi)
        self.val_dataset = build_from_cfg(cfg.dataset.val,DATASETS)
        self.test_dataset = build_from_cfg(cfg.dataset.test,DATASETS)
        self.vis_data = LoadImages("C:\\Users\\Admin\\Desktop\\yolox\\dataset\\images\\test", img_size=640)
        self.logger = build_from_cfg(self.cfg.logger,HOOKS,work_dir=self.work_dir)

        save_file = build_file(self.work_dir,prefix="config.yaml")
        save_cfg(save_file)

        self.iter = 0
        self.epoch = 0

        if self.max_epoch:
            if (self.train_dataset):
                self.total_iter = self.max_epoch * len(self.train_dataset)
            else:
                self.total_iter = 0
        else:
            self.total_iter = self.max_iter

        if (cfg.pretrained_weights):
            self.load(cfg.pretrained_weights, model_only=True)
        
        if self.resume_path is None:
            self.resume_path = search_ckpt(self.work_dir)
        if check_file(self.resume_path):
            self.resume()


    @property
    def finish(self):
        if self.max_epoch:
            return self.epoch>=self.max_epoch
        else:
            return self.iter>=self.max_iter
    
    def run(self):
        self.logger.print_log("Start running")
        
        while not self.finish:
            self.train()
            if check_interval(self.epoch,self.eval_interval):
                self.val()
            if check_interval(self.epoch,self.checkpoint_interval):
                self.save()
        self.test()

    def test_time(self):
        warmup = 10
        rerun = 100
        self.model.train()
        for batch_idx,(images,targets) in enumerate(self.train_dataset):
            break
        print("warmup...")
        for i in tqdm(range(warmup)):
            losses = self.model(images,targets)
            all_loss,losses = parse_losses(losses)
            self.optimizer.step(all_loss)
            self.scheduler.step(self.iter,self.epoch,by_epoch=True)
        jt.sync_all(True)
        print("testing...")
        start_time = time.time()
        for i in tqdm(range(rerun)):
            losses = self.model(images,targets)
            all_loss,losses = parse_losses(losses)
            self.optimizer.step(all_loss)
            self.scheduler.step(self.iter,self.epoch,by_epoch=True)
        jt.sync_all(True)
        batch_size = len(targets)*jt.world_size
        ptime = time.time()-start_time
        fps = batch_size*rerun/ptime
        print("FPS:", fps)

    def train(self):

        self.model.train()

        start_time = time.time()

        for batch_idx,(images,targets) in enumerate(self.train_dataset):

            losses = self.model(images,targets)
            all_loss,losses = parse_losses(losses)
            self.optimizer.step(all_loss)
            self.scheduler.step(self.iter,self.epoch,by_epoch=True)
            if check_interval(self.iter,self.log_interval) and self.iter>0:
                batch_size = len(images)*jt.world_size
                ptime = time.time()-start_time
                fps = batch_size*(batch_idx+1)/ptime
                eta_time = (self.total_iter-self.iter)*ptime/(batch_idx+1)
                eta_str = str(datetime.timedelta(seconds=int(eta_time)))
                data = dict(
                    name = self.cfg.name,
                    lr = self.optimizer.cur_lr(),
                    iter = self.iter,
                    epoch = self.epoch,
                    batch_idx = batch_idx,
                    batch_size = batch_size,
                    total_loss = all_loss,
                    fps=fps,
                    eta=eta_str
                )
                data.update(losses)
                data = sync(data)
                # is_main use jt.rank==0, so its scope must have no jt.Vars
                if jt.rank==0:
                    self.logger.log(data)
            
            self.iter+=1
            if self.finish:
                break
        self.epoch +=1


    @jt.no_grad()
    @jt.single_process_scope()
    def run_on_images(self,save_dir=None,**kwargs):
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        self.model.eval()
        for path, img, im0s, vid_cap in self.vis_data:
            img_jt = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img_jt = np.ascontiguousarray(img_jt)
            img_jt = jt.array(img_jt,dtype="float32") # uint8 to fp32
            img_jt /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img_jt.ndim == 3:
                img_jt = img_jt.unsqueeze(0)
            pred = self.model(img_jt,targets =1)
          
            pred_numpy = pred[0][0].numpy()
            a = 0
            for row in pred_numpy:
                x1, y1, x2, y2, score, label = row
                x1, y1, x2, y2, label = map(int, [x1, y1, x2, y2,label])
                if score<0.3:
                    continue
        # 选择颜色和线宽，可以根据标签或者其他条件动态调整
                color = (0, 255, 0)  # 这里以绿色为例
                thickness = 2
                a = a + 1
                # 在图像上画出矩形框
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                
                # 如果需要，还可以在框旁边添加标签和分数
                if label == 1:
                    text = f"nomask: {score:.2f}"
                else:
                    text = f"mask: {score:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 0.5, 2)[0]
                text_x = x1
                text_y = y1 - 5 if y1 - text_size[1] - 5 > 0 else y1 + text_size[1] + 5
                cv2.putText(img, text, (text_x, text_y), font, 0.5, color, 2)
            save__name = os.path.join("C:\\Users\\Admin\\Desktop\\JDet-master\\output",path.split("\\")[-1])
            print(save__name)
            if a > 0:
                cv2.imwrite(save__name,img)
            # print(pred[0][0].numpy())

        # Inference

        # for i,(images,targets) in tqdm(enumerate(self.test_dataset)):
        #     results = self.model(images,targets)
            
        #     # for t in targets:
        #     #     print(jt.attrs(t))
        #     #     print(jt.attrs(t["img_file"]))
        #     #     a = jt.vtos(t["img_file"])
        #     #     print(a)
        #     if save_dir:
        #         #visualize_results(sync(results),get_classes_by_name(self.test_dataset.dataset_type),[t["img_file"] for t in targets],save_dir, **kwargs)
        #         visualize_results(sync(results),get_classes_by_name('COCO'),[t["img_file"] for t in targets],save_dir, **kwargs)

    @jt.no_grad()
    @jt.single_process_scope()
    def val(self):
        if self.val_dataset is None:
            self.logger.print_log("Please set Val dataset")
        else:
            self.logger.print_log("Validating....")
            # TODO: need move eval into this function
            self.model.eval()
            #if model.is_training():
            #    model.eval()
            results = []
            for batch_idx,(images,targets) in tqdm(enumerate(self.val_dataset),total=len(self.val_dataset)):
                result = self.model(images,targets)
                results.extend([(r,t) for r,t in zip(sync(result),sync(targets))])
            eval_results = self.val_dataset.evaluate(results,self.work_dir,self.epoch,logger=self.logger)

            self.logger.log(eval_results,iter=self.iter)

    @jt.no_grad()
    @jt.single_process_scope()
    def test(self):

        if self.test_dataset is None:
            self.logger.print_log("Please set Test dataset")
        else:
            self.logger.print_log("Testing...")
            self.model.eval()
            results = []
            for batch_idx,(images,targets) in tqdm(enumerate(self.test_dataset),total=len(self.test_dataset)):
                result = self.model(images,targets)
                results.extend([(r,t) for r,t in zip(sync(result),sync(targets))])
                for mode in self.flip_test:
                    images_flip = images.copy()
                    if (mode == 'H'):
                        images_flip = images_flip[:, :, :, ::-1]
                    elif (mode == 'V'):
                        images_flip = images_flip[:, :, ::-1, :]
                    elif (mode == 'HV'):
                        images_flip = images_flip[:, :, ::-1, ::-1]
                    else:
                        assert(False)
                    result = self.model(images_flip,targets)
                    targets_ = copy.deepcopy(targets)
                    for i in range(len(targets_)):
                        targets_[i]["flip_mode"] = mode
                    results.extend([(r,t) for r,t in zip(sync(result),sync(targets_))])

            save_file = build_file(self.work_dir,f"test/test_{self.epoch}.pkl")
            pickle.dump(results,open(save_file,"wb"))
            if (self.cfg.dataset.test.type == "ImageDataset"):
                dataset_type = self.test_dataset.dataset_type
                data_merge_result(save_file,self.work_dir,self.epoch,self.cfg.name,dataset_type,self.cfg.dataset.test.images_dir)

    @jt.single_process_scope()
    def save(self):
        save_data = {
            "meta":{
                "jdet_version": jdet.__version__,
                "epoch": self.epoch,
                "iter": self.iter,
                "max_iter": self.max_iter,
                "max_epoch": self.max_epoch,
                "save_time":current_time(),
                "config": self.cfg.dump()
            },
            "model":self.model.state_dict(),
            "scheduler": self.scheduler.parameters(),
            "optimizer": self.optimizer.parameters()
        }
        save_file = build_file(self.work_dir,prefix=f"checkpoints/ckpt_{self.epoch}.pkl")
        jt.save(save_data,save_file)
        print("saved")
    
    def load(self, load_path, model_only=False):
        resume_data = jt.load(load_path)

        if (not model_only):
            meta = resume_data.get("meta",dict())
            self.epoch = meta.get("epoch",self.epoch)
            self.iter = meta.get("iter",self.iter)
            self.max_iter = meta.get("max_iter",self.max_iter)
            self.max_epoch = meta.get("max_epoch",self.max_epoch)
            self.scheduler.load_parameters(resume_data.get("scheduler",dict()))
            self.optimizer.load_parameters(resume_data.get("optimizer",dict()))
        if ("model" in resume_data):
            self.model.load_parameters(resume_data["model"])
        elif ("state_dict" in resume_data):
            self.model.load_parameters(resume_data["state_dict"])
        else:
            self.model.load_parameters(resume_data)

        self.logger.print_log(f"Loading model parameters from {load_path}")

    def resume(self):
        self.load(self.resume_path)