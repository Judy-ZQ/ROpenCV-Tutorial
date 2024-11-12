---
output: html_document
editor_options: 
  chunk_output_type: console
---


# 视频处理（视频IO以及Video模块）




## 读入视频文件并播放


``` r
#读取视频文件
cap = VideoCapture("video/vtest.avi")
#设置ret的值为TRUE
ret = TRUE
#设置引用变量，用来传递视频流读取的帧数据
frame = Mat()
#当视频流为打开状态且ret的值为TRUE是，循环播放视频
while(cap$isOpened() && ret){
  #读取视频流中的帧：
  #若读取失败（或者视频流结束），则ret的值为FALSE，循环终止，
  #若读取成功，则存入frame并显示在标题为frame的窗口中。
  #播放过程中按esc键，也会终止循环（即终止播放）
  ret = cap$read(frame)
  if(ret){
    cv_imshow('frame',frame)
    if(cv_waitKey(30)==27){
      break
    }
  }
}
cap$release()
```

可以通过帧率控制每帧显示时间


``` r
#读取视频文件
cap = VideoCapture("video/vtest.avi")

#获取视频帧率
fps = cap$get(CAP_PROP_FPS)
#设置两帧之间的间隔
pauseTime = 1000/fps

#设置ret的值为TRUE
ret = TRUE
#设置引用变量，用来传递视频流读取的帧数据
frame = Mat()
#当视频流为打开状态且ret的值为TRUE是，循环播放视频
while(cap$isOpened() && ret){
  #读取视频流中的帧，若读取失败（或者视频流结束），则ret的值为FALSE，循环终止
  #若读取成功，则存入frame并显示在标题为frame的窗口中
  #播放过程中按esc键，也会终止循环（即终止播放）
  ret = cap$read(frame)
  if(ret){
    cv_imshow('frame',frame)
    if(cv_waitKey(pauseTime)==27){
      break
    }
  }
}
cap$release()
```



## 视频相似性度量

### PSNR(Peak signal-to-noise ratio)方法

PSNR是使用“局部均值误差”来判断视频差异的最简单的方法，相应公式为：

$$
\begin{aligned}
MSE &= \frac{1}{c\times m\times n} \sum(I_1 - I_2)^2 \\
PSNR &= 10 \times lg \left( \frac{MAX_I^2}{MSE} \right)
\end{aligned}
$$

其中：

$I_1$和$I_2$为待比较的两个图像，它们的行、列数分别是$m$，$n$，有$c$个通道，$MAX_I$表示图像像素的最大可能取值，当图像深度为8位无符号整数时，那么$MAX_I$就是255。


``` r
getPSNR = function(I1, I2)
{
  psnr = NA
  I1_arr = I1$cv2r()
  I2_arr = I2$cv2r()
  s = (abs(I1_arr - I2_arr)) ^ 2
  sse = sum(s) # sum channels
  if (sse <= 1e-10){
    # for small values return zero
    psnr = Inf
  }else{
    mse  = sse / prod(dim(I1_arr))
    psnr = 10.0 * log10((255 * 255) / mse) #限定为8bit图像？
  }
  return(psnr)
}
# ![get-psnr]

# 单通道图像测试
img1_mat = matrix(1:9,nr=3,nc=3)
img1 = Mat(3,3,CV_8UC1)
img1$r2cv(img1_mat)
getPSNR(img1,img1)
```

```
## [1] Inf
```

``` r
img2_mat = img1_mat
img2_mat[2,2] = 0
img2 = Mat(3,3,CV_8UC1)
img2$r2cv(img2_mat)
getPSNR(img1,img2)
```

```
## [1] 43.69383
```

``` r
mse = 5^2/9
10*log10(255^2/mse)
```

```
## [1] 43.69383
```

``` r
# 3通道彩色图像测试
img1_arr = array(0,dim=c(3,3,3))
img2_arr = array(0,dim=c(3,3,3))
img1_arr[,,1] = img1_mat
img1_arr[,,2] = img1_mat
img1_arr[,,3] = img1_mat
img2_arr[,,1] = img2_mat
img2_arr[,,2] = img2_mat
img2_arr[,,3] = img2_mat
img1 = Mat(3,3,CV_8UC3)
img1$r2cv(img1_mat)
img2 = Mat(3,3,CV_8UC3)
img2$r2cv(img2_mat)
getPSNR(img1,img2)
```

```
## [1] 43.69383
```

``` r
mse = 3*5^2/(3*9)
10*log10(255^2/mse)
```

```
## [1] 43.69383
```


### SSIM

**示例***


``` r
# ![get-mssim] 用R矩阵实现
getMSSIM = function(i1, i2)
{
    C1 = 6.5025
    C2 = 58.5225
    #/***************************** INITS **********************************/
    
    i1_arr = i1$cv2r()
    i2_arr = i2$cv2r()
    
    chn = ifelse(length(dim(i1_arr))>2,dim(i1_arr)[3],1)
    
    i1_2_arr = i1_arr^2
    i1_2 = Mat(i1$rows,i1$cols,i1$type())
    i1_2$r2cv(i1_2_arr)
    i1_2$convertTo(i1_2,CV_32F)
      
    i2_2_arr = i2_arr^2
    i2_2 = Mat(i2$rows,i2$cols,i2$type())
    i2_2$r2cv(i2_2_arr)
    i2_2$convertTo(i2_2,CV_32F)
    
    i1_i2_arr = i1_arr * i2_arr
    i1_i2 = Mat(i1$rows,i1$cols,i1$type())
    i1_i2$r2cv(i1_i2_arr)
    i1_i2$convertTo(i1_i2,CV_32F)
    
    #*************************** END INITS **********************************/
    mu1 = Mat()
    mu2 = Mat()                   # PRELIMINARY COMPUTING
    cv_GaussianBlur(i1, mu1, Size(11, 11), 1.5)
    cv_GaussianBlur(i2, mu2, Size(11, 11), 1.5)

    mu1_2_arr = mu1$cv2r()^2
    mu2_2_arr = mu2$cv2r()^2
    mu1_mu2_arr = mu1$cv2r() * mu2$cv2r()
    
    
    sigma1_2 = Mat()
    sigma2_2 = Mat()
    sigma1_sigma2 = Mat()
    
    cv_GaussianBlur(i1_2, sigma1_2, Size(11, 11), 1.5)
    cv_GaussianBlur(i2_2, sigma2_2, Size(11, 11), 1.5)
    cv_GaussianBlur(i1_i2, sigma1_sigma2, Size(11, 11), 1.5)
    
    sigma1_2_arr = sigma1_2$cv2r()
    sigma2_2_arr = sigma2_2$cv2r()
    sigma1_sigma2_arr = sigma1_sigma2$cv2r()
    
    sigma1_2_arr = sigma1_2_arr - mu1_2_arr
    sigma2_2_arr = sigma2_2_arr - mu2_2_arr
    sigma1_sigma2_arr = sigma1_sigma2_arr - mu1_mu2_arr
    
    t1 = 2*mu1_mu2_arr + C1
    t2 = 2*sigma1_sigma2_arr + C2
    t3 = t1 * t2
    
    t1 = mu1_2_arr + mu2_2_arr + C1
    t2 = sigma1_2_arr + sigma2_2_arr + C2
    t4 = t1 * t2
    ssimg_map = t3 / t4
    if(chn==1){
      mssim = mean(ssimg_map)
    }else{
      mssim = apply(ssimg_map,3,mean)
    }
    return (mssim)
}
# ![get-mssim]

# 单通道图像测试
img1_mat = matrix(1:9,nr=3,nc=3)
img1 = Mat(3,3,CV_8UC1)
img1$r2cv(img1_mat)
getMSSIM(img1,img1)
```

```
## [1] 1
```

``` r
img2_mat = img1_mat
img2_mat[2,2] = 0
img2 = Mat(3,3,CV_8UC1)
img2$r2cv(img2_mat)
getMSSIM(img1,img2)
```

```
## [1] 0.9066809
```

``` r
# 3通道彩色图像测试
img1_arr = array(0,dim=c(3,3,3))
img1_arr[,,1] = img1_mat
img1_arr[,,2] = img1_mat
img1_arr[,,3] = img1_mat
img1 = Mat(3,3,CV_8UC3)
img1$r2cv(img1_arr)

img2_arr = array(0,dim=c(3,3,3))
img2_arr[,,1] = img2_mat
img2_arr[,,2] = img2_mat
img2_arr[,,3] = img2_mat
img2 = Mat(3,3,CV_8UC3)
img2$r2cv(img2_arr)

getMSSIM(img1,img2)
```

```
## [1] 0.9066809 0.9066809 0.9066809
```

**示例**


``` r
sourceReference ="video/Megamind01.avi"
captRefrnc = VideoCapture(sourceReference)
sourceCompareWith = "video/Megamind_bugy01.avi"
captUndTst = VideoCapture(sourceCompareWith)

#captRefrnc$open(sourceReference)

if (!captRefrnc$isOpened()){
  cat("Could not open reference " , sourceReference , '\n')
}
if (!captUndTst$isOpened()){
  cat("Could not open case test " , sourceCompareWith , '\n')
}

captRefrnc$get(CAP_PROP_FRAME_WIDTH)
captUndTst$get(CAP_PROP_FRAME_WIDTH)

captRefrnc$get(CAP_PROP_FRAME_HEIGHT)
captUndTst$get(CAP_PROP_FRAME_HEIGHT)


frame1 = Mat()
frame2 = Mat()
frameNum = -1
psnrTriggerValue = 35
while(1){
  ret1 = captRefrnc$read(frame1)
  ret2 = captUndTst$read(frame2)
  if(!(ret1 && ret2)){
    break
  }
  frameNum = frameNum+1
  psnrV = getPSNR(frame1,frame2)
  
  res = paste0("Frame: ",frameNum," ", psnrV,"dB")
  if(psnrV && psnrV < psnrTriggerValue){
    mssimV = getMSSIM(frame1, frame2)
    res = paste0(res,"\tMSSIM: R ",round(mssimV[3]*100,2),"% G ",round(mssimV[2]*100,2),"% B ",round(mssimV[1]*100,2),"%")
  }
  res = paste0(res,"\n")
  cat(res)
  
  cv_imshow("参考视频",frame1)
  cv_imshow("检测视频",frame2)
  
  cc = cv_waitKey(30)
  if (27 == cc) {
    break
  }
}
# 感觉有些停滞
```

## 背景减除

背景减除（BS）是通过使用静态相机生成前景遮罩（即，包含属于场景中的运动对象的像素的二值图像）的常见且广泛使用的技术。

顾名思义，BS计算前景遮罩，在当前帧和背景模型之间执行减法运算，其中包含场景的静态部分，或者更一般地说，根据所观察场景的特征，可以认为是背景的所有内容。

![](images/tutorial/Background_Subtraction_Tutorial_Scheme.png)

OpenCV中的背景减除函数为：

```
createBackgroundSubtractorMOG2(
  int  history = 500, 
  double  varThreshold = 16,
  bool  detectShadows = true 
)
```


参数解释如下：

* history表示过往帧数，500帧，选择history = 1就变成两帧差。
* varThreshold表示像素与模型之间的马氏距离，值越大，只有那些最新的像素会被归到前景，值越小前景对光照越敏感。
* detectShadows 是否保留阴影检测，请选择False这样速度快点。

**示例**


``` r
#创建高斯混合模型分离算法器mog2：
#history表示过往帧数，选择history = 1就变成两帧差；
#varThreshold表示像素与模型之间的马氏距离，值越大，只有那些最新的像素会被归到前景，值越小前景对光照越敏感；
#detectShadows 是否保留阴影检测，请选择False这样速度快点。
mog2 = BackgroundSubtractorMOG2(history = 600, varThreshold = 500, detectShadows = FALSE)
#读取视频文件
cap = VideoCapture("video/vtest.avi")
#获取视频帧率
fps = cap$get(CAP_PROP_FPS)
#设置两帧之间的间隔
pauseTime = 1000/fps

# #生成图像矩阵的引用类变量
# tmp.ref = encloseRef(cv.mat.Mat01())

#定义getPerson函数，用于提取视频中的人物
getPerson = function(frame){
  #获取前景遮罩
  mask = Mat()
  mog2$apply(frame,mask,-1)
  
  #去除噪声
  line_ele = cv_getStructuringElement(MORPH_RECT,Size(1,5))
  cv_morphologyEx(mask,mask,MORPH_OPEN,line_ele)
  cv_imshow("mask",mask)
  
  #检测轮廓
  contours = stdVecOfVecOfPoint()
  hierarchy = stdVecOfVec4i()
  cv_findContours(mask,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE)
  
  for(i in 1:contours$size()-1){
    area = cv_contourArea(contours[[i]])
    if(area<150){
      next
    }
    rct = cv_minAreaRect(contours[[i]])
    cv_ellipse(frame,rct,Scalar(0,255,0),2)
    
    cv_circle(frame,rct$center,2,Scalar(255,0,0),2)
  }
}

#设置ret的值为TRUE
ret = TRUE
#设置引用变量，用来传递视频流读取的帧数据
frame = Mat()
#当视频流为打开状态且ret的值为TRUE是，循环播放视频
while(cap$isOpened() && ret){
  #读取视频流中的帧，若读取失败（或者视频流结束），则ret的值为FALSE，循环终止
  #若读取成功，则存入frame并显示在标题为frame的窗口中
  #播放过程中按esc键，也会终止循环（即终止播放）
  ret = cap$read(frame)
  if(ret){
    getPerson(frame)
    
    cv_imshow('frame',frame)
    if(cv_waitKey(30)==27){
      break
    }
  }
}
```



## 光流


主要流程：

* 加载视频；
* 调用GoodFeaturesToTrack 函数寻找兴趣点（关键点）；
* 调用CalcOpticalFlowPyrLK 函数计算出两帧图像中兴趣点的移动情况；
* 删除未移动的兴趣点；
* 在两次移动的点之间绘制一条线段。

其中：calcOpticalFlowFarneback函数的参数如下：
        
*	prev：前一帧图片 
*	next：下一帧图片，格式与prev相同 
*	flow：一个CV_32FC2格式的光流图 
*	pyr_scale： 构建图像金字塔尺度 
*	levels： 图像金字塔层数 
*	winsize： 窗口尺寸，值越大探测高速运动的物体越容易，但是越模糊，同时对噪声的容错性越强 
*	iterations：对每层金字塔的迭代次数 
*	poly_n：每个像素中找到多项式展开的邻域像素的大小，值越大越光滑，也越稳定
*	poly_sigma：高斯标准差，用来平滑倒数，poly_n越大，poly_sigma应该适当增加
*	flags：光流的方式，有OPTFLOW_USE_INITIAL_FLOW 和OPTFLOW_FARNEBACK_GAUSSIAN 两种

**示例**


``` r
#读取视频文件
#cap = cv.VideoCapture("video/slow.flv")
cap = VideoCapture("video/vtest.avi")
#生成随机矩阵，每行表示一个随机颜色
colorMat = matrix(rep(sample(0:255,3,replace=T),100),nr=100,nc=3)

#读取第一帧
old_frame = Mat()
ret = cap$read(old_frame)
if(!ret){
  stop("error to read video")
}

#将帧灰度化
old_gray = Mat()
cv_cvtColor(old_frame,old_gray,COLOR_BGR2GRAY)


#获取特征点
p0 = stdVecOfPoint2f()
cv_goodFeaturesToTrack(old_gray,p0,100,0.1,7,blockSize = 7)
good_ini = p0

#定义计算两点距离的函数
caldist = function(x1,y1,x2,y2){
  return(abs(x2-x1)+abs(y2-y1))
}

#生成一个mask
old_frame_info = rcv_matInfo(old_frame)
mask = Mat(old_frame_info$height,old_frame_info$width,old_frame_info$type)

#光流跟踪
while(cap$isOpened() && ret){
  #读取视频流中的帧，若读取失败（或者视频流结束），则ret的值为FALSE，循环终止
  #若读取成功，则存入frame并显示在标题为frame的窗口中
  #播放过程中按esc键，也会终止循环（即终止播放）
  frame = Mat()
  ret = cap$read(frame)
  if(ret){
    tryCatch({
      frame_gray = Mat()
      cv_cvtColor(frame,frame_gray,COLOR_BGR2GRAY)
      
      p1 = stdVecOfPoint2f()
      status = stdVecOfuchar()
      err = stdVecOffloat()
      cv_calcOpticalFlowPyrLK(old_gray,frame_gray,p0,p1,status,err,Size(15,15),2,TermCriteria(3,10,0.02))
      
      #status==1
      good_new = stdVecOfPoint2f()
      good_old = stdVecOfPoint2f()
      for(i in 1:status$size()-1){
        if(as.numeric(status[[i]])==1){
          good_new$push_back(p1[[i]])
          good_old$push_back(p0[[i]])
        }
      }
      
      #删除静止点
      k=0
      good_new1 = stdVecOfPoint2f()
      good_old1 = stdVecOfPoint2f()
      good_ini1 = stdVecOfPoint2f()
      while(k<good_new$size()){
        ddist = caldist(good_new[[k]]$x,good_new[[k]]$y,good_old[[k]]$x,good_old[[k]]$y)
        if(ddist<=2){
        }else{
          good_ini1$push_back(good_ini[[k]])
          good_old1$push_back(good_old[[k]])
          good_new1$push_back(good_new[[k]])
        }
        k = k+1
      }
      
      #绘制跟踪线
      for(i in 1:good_new1$size()-1){
        sca = Scalar(colorMat[i+1,1],colorMat[i+1,2],colorMat[i+1,3])
        cv_line(mask,Point(good_new1[[i]]$x,good_new1[[i]]$y),Point(good_old1[[i]]$x,good_old1[[i]]$y),sca)
        cv_circle(frame,Point(good_new1[[i]]$x,good_new1[[i]]$y),5,sca,-1)
      }
      img = Mat()
      cv_add(frame,mask,img)
      
      cv_imshow('frame',img)
      if(cv_waitKey(30)==27){
        break
      }
      
      #更新
      old_gray = frame_gray$clone()
      p0 = good_new1
      
      if(good_ini1$size()<40){
        cv_goodFeaturesToTrack(old_gray,good_ini,100,0.1,7,blockSize = 7)
      }
    },error=function(e){
      
    })
  }
}
cap$release()
```


**示例**


``` r
#读取视频文件
cap = VideoCapture("video/vtest.avi")

#读取第一帧
frame1 = Mat()
ret = cap$read(frame1)
frame1_info = rcv_matInfo(frame1)
hsv_mat = array(0,dim=c(frame1_info$height,frame1_info$width,frame1_info$channels))
hsv_mat[,,2] = 255

prvs = Mat()
cv_cvtColor(frame1,prvs,COLOR_BGR2GRAY)

while(cap$isOpened() && ret){
  frame2 = Mat()
  ret = cap$read(frame2)
  
  nxt = Mat()
  cv_cvtColor(frame2,nxt,COLOR_BGR2GRAY)
  
  #返回一个两通道的光流向量，实际上是每个点的像素位移值
  flow = Mat()
  cv_calcOpticalFlowFarneback(prvs,nxt,flow,0.5,3,15,3,5,1.2,0)
  
  planes = stdVecOfMat()
  cv_split(flow,planes)
  mag = Mat()
  angle = Mat()
  cv_cartToPolar(planes[[0]],planes[[1]],mag,angle)
  cv_normalize(mag,mag,0,255,NORM_MINMAX)
  
  mag_mat = mag$cv2r()
  angle_mat = angle$cv2r()
  
  hsv_mat[,,1] = round(angle_mat*180/pi)
  hsv_mat[,,3] = mag_mat
  # hsv = cv.mat.r2cv(hsv.mat,"CV_32FC3")
  hsv = Mat(nrow(hsv_mat),ncol(hsv_mat),frame1_info$type)
  hsv$r2cv(hsv_mat)
  bgr = Mat()
  cv_cvtColor(hsv,bgr,COLOR_HSV2BGR)

  cv_imshow('frame2',bgr)
  cv_imshow('frame1',frame2)
  
  if(cv_waitKey(30)==27){
      break
    }
}

cap$release()
```


