#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv/cv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <Windows.h>
//#include <vector>

using namespace std;
using namespace cv;

Mat OTSU(cv::Mat srcGray);



int main()
{
	//读图
	Mat RGBImage = imread("HU.jpg");
	//Mat BGRImage = imread("rgb.bmp");
	double Start = GetTickCount();
	//imshow("RGB", RGBImage);
	Mat BGRImage;
	    /***************************************************************/
	   /***************************************************************/
	  /********************加快图像处理，减小时间**********************/
	 /***************************************************************/
	/***************************************************************/

	/*************************高斯金字塔**************************/
	pyrDown(RGBImage, BGRImage, Size(RGBImage.cols / 2, RGBImage.rows / 2));
	//resize(RGBImage, BGRImage, Size(RGBImage.cols/2,RGBImage.rows/2), 0, 0, CV_INTER_LINEAR);

/**********************////////////////////////////////////********************************************/
	/*##########??????
   //得到H、S、I三分量****************************************************
	Mat Hvalue, Svalue, Ivalue, HSIImage;
	HSIImage = Mat(Size(BGRImage.cols, BGRImage.rows), CV_8UC3);

	vector <Mat> channels;
	split(HSIImage, channels);
	Hvalue = channels.at(0);
	Svalue = channels.at(1);
	Ivalue = channels.at(2);

	for (int i = 0; i < BGRImage.rows; ++i)
		for (int j = 0; j < BGRImage.cols; ++j)
		{
			double H, S, I;
			int Bvalue = BGRImage.at<Vec3b>(i, j)(0);
			int Gvalue = BGRImage.at<Vec3b>(i, j)(1);
			int Rvalue = BGRImage.at<Vec3b>(i, j)(2);

			//求Theta =acos((((Rvalue - Gvalue) + (Rvalue - Bvalue)) / 2) / sqrt(pow((Rvalue - Gvalue), 2) + (Rvalue - Bvalue)*(Gvalue - Bvalue)));
			double numerator = ((Rvalue - Gvalue) + (Rvalue - Bvalue)) / 2;
			double denominator = sqrt(pow((Rvalue - Gvalue), 2) + (Rvalue - Bvalue)*(Gvalue - Bvalue));
			if (denominator == 0) H = 0;
			else {
				double Theta = acos(numerator / denominator) * 180 / 3.14;
				if (Bvalue <= Gvalue)
					H = Theta;
				else  H = 360 - Theta;
			}
			Hvalue.at<uchar>(i, j) = (int)(H * 255 / 360); //为了显示将[0~360]映射到[0~255]

														   //求S = 1-3*min(Bvalue,Gvalue,Rvalue)/(Rvalue+Gvalue+Bvalue);
			int minvalue = Bvalue;
			if (minvalue > Gvalue) minvalue = Gvalue;
			if (minvalue > Rvalue) minvalue = Rvalue;
			numerator = 3 * minvalue;
			denominator = Rvalue + Gvalue + Bvalue;
			if (denominator == 0)  S = 0;
			else {
				S = 1 - numerator / denominator;
			}
			Svalue.at<uchar>(i, j) = (int)(S * 255);//为了显示将[0~1]映射到[0~255]

			I = (Rvalue + Gvalue + Bvalue) / 3;
			Ivalue.at<uchar>(i, j) = (int)(I);
		}
	//merge(channels, HSIImage);
	//imshow("H", Hvalue);



	//OSTU算法**************************************************************
	Mat H_OTSU = OTSU(Hvalue);
	Mat S_OTSU = OTSU(Svalue);
	Mat I_OTSU = OTSU(Ivalue);
	//imshow("OSTU", S_OTSU);
	
	//HSI融合***************************************************************
	
	//H、S、I位与，求公共部分
	Mat Fusion = H_OTSU&S_OTSU&I_OTSU;
	//imshow("F", Fusion);
	//膨胀，改善效果
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
	Mat Mix;
	dilate(Fusion, Mix, element);
	//imshow("M", Mix);


	*****##########????????******/
/*********************///////////////////////////******************************************/
	//**********************************RGB转Lab****************************
	//得到Lab图像a通道
	vector<Mat>channels1;
	Mat Lab;
	cvtColor(BGRImage, Lab, CV_RGB2Lab);
	split(Lab, channels1);
	Mat A = channels1.at(1);
	//imshow("A", A);



	//*********************************K-means算法****************************
	Mat img = A;
	//生成一维图像采样点，包括所有图像像素点，注意采样格式为32bit浮点数
	Mat sample(img.cols*img.rows, 1, CV_32FC1);

	//标记矩阵，32位整形
	Mat labels(img.cols*img.rows, 1, CV_32SC1);
	uchar *p;
	int i, j, k = 0;

	for (i = 0; i < img.rows; i++)
	{
		p = img.ptr<uchar>(i);

		for (j = 0; j < img.cols; j++)
		{
			sample.at<Vec3f>(k, 0)[0] = float(p[j]);
			//sample.at<Vec3f>(k, 0)[1] = float(p[j * 3 + 1]);
			//sample.at<Vec3f>(k, 0)[2] = float(p[j * 3 + 2]);
			//A.at<uchar>(i,j)= float(p[j * 3 + 1]);
			k++;

		}

	}

	//对A通道滤波（不可行）
	//vector<Mat>channels;
	////cvtColor(srcImage, dstImage, CV_RGB2Lab);
	//split(sample, channels);
	//Mat srcImage = channels.at(1);
	//
	////Mat srcImage = sample.channels[1];
	//Mat sample1;
	//Mat ele = getStructuringElement(MORPH_RECT, Size(3, 3));
	//morphologyEx(srcImage, sample1, MORPH_BLACKHAT, ele);

	int clusterCount = 2;
	Mat centers(clusterCount, 1, sample.type());
	kmeans(sample, clusterCount, labels,
		TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
		3, KMEANS_PP_CENTERS, centers);


	//我们已知有两个个聚类，用不同的灰度层表示
	Mat img1(img.rows, img.cols, CV_8UC1);
	float step = 255 / (clusterCount - 1);
	k = 0;
	for (i = 0; i < img1.rows; i++)
	{
		p = img1.ptr<uchar>(i);
		for (j = 0; j < img1.cols; j++)
		{
			int tt = labels.at<int>(k, 0);
			k++;
			p[j] = 255 - tt*step;
		}
	}

	//imshow("Kmeans", img1);


	//************************Kmeans效果处理（闭运算）*********************
	//去除孤立点
	Mat element2 = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
	Mat Open_Mat;
	morphologyEx(img1, Open_Mat, MORPH_CLOSE, element2);
	//imshow("R", Open_Mat);

	//***********************************************************求质心
	
	//////////////////////仅仅使用A通道的图像，其它未用

	//Mat src = Open_Mat;
	Mat src_gray= Open_Mat;
	//double area;
	int thresh = 30;
	int max_thresh = 255;
	//cvtColor(src, src_gray, CV_BGR2GRAY);//灰度化 	
	//GaussianBlur(src, src, Size(3, 3), 0.1, 0, BORDER_DEFAULT);
	//blur(src_gray, src_gray, Size(3, 3)); //滤波 	
	//namedWindow("image", CV_WINDOW_AUTOSIZE);
	//imshow("image", src);
	//moveWindow("image", 20, 20);
	//定义Canny边缘检测图像 	
	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//利用canny算法检测边缘 	
	Canny(src_gray, canny_output, thresh, thresh * 3, 3);
	/*namedWindow("canny", CV_WINDOW_AUTOSIZE);
	imshow("canny", canny_output);
	moveWindow("canny", 550, 20);*/
	//查找轮廓 	
	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	//计算轮廓矩 	
	vector<Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}
	//计算轮廓的质心 	
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mc[i] = Point2d(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}
	//画轮廓及其质心并显示 	
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(255, 0, 0);
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		circle(drawing, mc[i], 5, Scalar(0, 0, 255), -1, 8, 0);
		rectangle(drawing, boundingRect(contours.at(i)), cvScalar(0, 255, 0));
		// 计算矩阵面积
		//area = contourArea(contours.at(i));
		//cout << area << endl;
		char tam[100];
		sprintf(tam, "(%0.0f,%0.0f)", mc[i].x, mc[i].y);
		/*if (area>1000&&area<1700)
		    cout << Point(mc[i].x, mc[i].y) << endl;*/
		putText(drawing, tam, Point(mc[i].x, mc[i].y), FONT_HERSHEY_SIMPLEX, 0.4, cvScalar(255, 0, 255), 1);

	}
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
	double myends = GetTickCount();
	cout << "时间" << myends - Start << endl;
	//moveWindow("Contours", 1100, 20);
	
	



	waitKey(0);
	//src.release();
	src_gray.release();
	return 0;

}




Mat OTSU(cv::Mat srcGray)
{
	//灰度转换
	//Mat srcGray;
	//cvtColor(srcImage, srcGray, CV_RGB2GRAY);
	//imshow("srcGray", srcGray);

	int nCols = srcGray.cols;
	int nRows = srcGray.rows;
	int threshold = 0;

	//初始化统计参数
	int nSumPix[256] = { 0 };
	float nProDis[256] = { 0 };
	/*for (int i = 0; i < 256; i++)
	{
	nSumPix[i] = 0;
	nProDis[i] = 0;

	}*/

	//统计灰度图级中每个像素在整幅图像中的个数
	for (int i = 0; i < nCols; i++)
	{
		for (int j = 0; j < nRows; j++)
		{
			nSumPix[(int)srcGray.at<uchar>(j, i)]++;
		}
	}

	//计算每个灰度级占图像中的概率分布
	for (int i = 0; i < 256; i++)
	{
		nProDis[i] = (float)nSumPix[i] / (nCols*nRows);
	}

	//遍历灰度级【0，255】，计算出最大类间方差下的阈值
	float w0, w1, u0_temp, u1_temp, u0, u1, delta_temp;
	double delta_max = 0.0;
	for (int i = 0; i < 256; i++)
	{
		//初始化相关参数
		w0 = w1 = u0_temp = u1_temp = u0 = u1 = delta_temp = 0;
		for (int j = 0; j < 256; j++)
		{
			//背景部分
			if (j <= i)
			{
				//当前i为分割阈值，第一类总的概率
				w0 += nProDis[j];
				u0_temp += j*nProDis[j];
			}
			//前景部分
			else
			{
				//当前i为分割阈值，第一类总的概率
				w1 += nProDis[j];
				u1_temp += j*nProDis[j];
			}
		}
		//分别计算各类的平均灰度
		u0 = u0_temp / w0;
		u1 = u1_temp / w1;
		delta_temp = (float)(w0*w1*pow((u0 - u1), 2));
		//依次找到最大类间方差下的阈值
		if (delta_temp > delta_max)
		{
			delta_max = delta_temp;
			threshold = i;
		}
	}

	cout << threshold << endl;
	//定义输出结果图像
	Mat otsuResultImage = Mat::zeros(srcGray.rows, srcGray.cols, CV_8UC1);
	//利用得到的阈值实现二值化操作
	for (int i = 0; i < srcGray.rows; i++)
	{
		for (int j = 0; j < srcGray.cols; j++)
		{
			if (srcGray.at<uchar>(i, j) > threshold)
			{
				otsuResultImage.at<uchar>(i, j) = 255;
			}
			else
			{
				otsuResultImage.at<uchar>(i, j) = 0;
			}
		}
	}
	return otsuResultImage;
}