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
	//��ͼ
	Mat RGBImage = imread("HU.jpg");
	//Mat BGRImage = imread("rgb.bmp");
	double Start = GetTickCount();
	//imshow("RGB", RGBImage);
	Mat BGRImage;
	    /***************************************************************/
	   /***************************************************************/
	  /********************�ӿ�ͼ������Сʱ��**********************/
	 /***************************************************************/
	/***************************************************************/

	/*************************��˹������**************************/
	pyrDown(RGBImage, BGRImage, Size(RGBImage.cols / 2, RGBImage.rows / 2));
	//resize(RGBImage, BGRImage, Size(RGBImage.cols/2,RGBImage.rows/2), 0, 0, CV_INTER_LINEAR);

/**********************////////////////////////////////////********************************************/
	/*##########??????
   //�õ�H��S��I������****************************************************
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

			//��Theta =acos((((Rvalue - Gvalue) + (Rvalue - Bvalue)) / 2) / sqrt(pow((Rvalue - Gvalue), 2) + (Rvalue - Bvalue)*(Gvalue - Bvalue)));
			double numerator = ((Rvalue - Gvalue) + (Rvalue - Bvalue)) / 2;
			double denominator = sqrt(pow((Rvalue - Gvalue), 2) + (Rvalue - Bvalue)*(Gvalue - Bvalue));
			if (denominator == 0) H = 0;
			else {
				double Theta = acos(numerator / denominator) * 180 / 3.14;
				if (Bvalue <= Gvalue)
					H = Theta;
				else  H = 360 - Theta;
			}
			Hvalue.at<uchar>(i, j) = (int)(H * 255 / 360); //Ϊ����ʾ��[0~360]ӳ�䵽[0~255]

														   //��S = 1-3*min(Bvalue,Gvalue,Rvalue)/(Rvalue+Gvalue+Bvalue);
			int minvalue = Bvalue;
			if (minvalue > Gvalue) minvalue = Gvalue;
			if (minvalue > Rvalue) minvalue = Rvalue;
			numerator = 3 * minvalue;
			denominator = Rvalue + Gvalue + Bvalue;
			if (denominator == 0)  S = 0;
			else {
				S = 1 - numerator / denominator;
			}
			Svalue.at<uchar>(i, j) = (int)(S * 255);//Ϊ����ʾ��[0~1]ӳ�䵽[0~255]

			I = (Rvalue + Gvalue + Bvalue) / 3;
			Ivalue.at<uchar>(i, j) = (int)(I);
		}
	//merge(channels, HSIImage);
	//imshow("H", Hvalue);



	//OSTU�㷨**************************************************************
	Mat H_OTSU = OTSU(Hvalue);
	Mat S_OTSU = OTSU(Svalue);
	Mat I_OTSU = OTSU(Ivalue);
	//imshow("OSTU", S_OTSU);
	
	//HSI�ں�***************************************************************
	
	//H��S��Iλ�룬�󹫹�����
	Mat Fusion = H_OTSU&S_OTSU&I_OTSU;
	//imshow("F", Fusion);
	//���ͣ�����Ч��
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
	Mat Mix;
	dilate(Fusion, Mix, element);
	//imshow("M", Mix);


	*****##########????????******/
/*********************///////////////////////////******************************************/
	//**********************************RGBתLab****************************
	//�õ�Labͼ��aͨ��
	vector<Mat>channels1;
	Mat Lab;
	cvtColor(BGRImage, Lab, CV_RGB2Lab);
	split(Lab, channels1);
	Mat A = channels1.at(1);
	//imshow("A", A);



	//*********************************K-means�㷨****************************
	Mat img = A;
	//����һάͼ������㣬��������ͼ�����ص㣬ע�������ʽΪ32bit������
	Mat sample(img.cols*img.rows, 1, CV_32FC1);

	//��Ǿ���32λ����
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

	//��Aͨ���˲��������У�
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


	//������֪�����������࣬�ò�ͬ�ĻҶȲ��ʾ
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


	//************************KmeansЧ�����������㣩*********************
	//ȥ��������
	Mat element2 = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
	Mat Open_Mat;
	morphologyEx(img1, Open_Mat, MORPH_CLOSE, element2);
	//imshow("R", Open_Mat);

	//***********************************************************������
	
	//////////////////////����ʹ��Aͨ����ͼ������δ��

	//Mat src = Open_Mat;
	Mat src_gray= Open_Mat;
	//double area;
	int thresh = 30;
	int max_thresh = 255;
	//cvtColor(src, src_gray, CV_BGR2GRAY);//�ҶȻ� 	
	//GaussianBlur(src, src, Size(3, 3), 0.1, 0, BORDER_DEFAULT);
	//blur(src_gray, src_gray, Size(3, 3)); //�˲� 	
	//namedWindow("image", CV_WINDOW_AUTOSIZE);
	//imshow("image", src);
	//moveWindow("image", 20, 20);
	//����Canny��Ե���ͼ�� 	
	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//����canny�㷨����Ե 	
	Canny(src_gray, canny_output, thresh, thresh * 3, 3);
	/*namedWindow("canny", CV_WINDOW_AUTOSIZE);
	imshow("canny", canny_output);
	moveWindow("canny", 550, 20);*/
	//�������� 	
	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	//���������� 	
	vector<Moments> mu(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mu[i] = moments(contours[i], false);
	}
	//�������������� 	
	vector<Point2f> mc(contours.size());
	for (int i = 0; i < contours.size(); i++)
	{
		mc[i] = Point2d(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
	}
	//�������������Ĳ���ʾ 	
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		Scalar color = Scalar(255, 0, 0);
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
		circle(drawing, mc[i], 5, Scalar(0, 0, 255), -1, 8, 0);
		rectangle(drawing, boundingRect(contours.at(i)), cvScalar(0, 255, 0));
		// ����������
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
	cout << "ʱ��" << myends - Start << endl;
	//moveWindow("Contours", 1100, 20);
	
	



	waitKey(0);
	//src.release();
	src_gray.release();
	return 0;

}




Mat OTSU(cv::Mat srcGray)
{
	//�Ҷ�ת��
	//Mat srcGray;
	//cvtColor(srcImage, srcGray, CV_RGB2GRAY);
	//imshow("srcGray", srcGray);

	int nCols = srcGray.cols;
	int nRows = srcGray.rows;
	int threshold = 0;

	//��ʼ��ͳ�Ʋ���
	int nSumPix[256] = { 0 };
	float nProDis[256] = { 0 };
	/*for (int i = 0; i < 256; i++)
	{
	nSumPix[i] = 0;
	nProDis[i] = 0;

	}*/

	//ͳ�ƻҶ�ͼ����ÿ������������ͼ���еĸ���
	for (int i = 0; i < nCols; i++)
	{
		for (int j = 0; j < nRows; j++)
		{
			nSumPix[(int)srcGray.at<uchar>(j, i)]++;
		}
	}

	//����ÿ���Ҷȼ�ռͼ���еĸ��ʷֲ�
	for (int i = 0; i < 256; i++)
	{
		nProDis[i] = (float)nSumPix[i] / (nCols*nRows);
	}

	//�����Ҷȼ���0��255��������������䷽���µ���ֵ
	float w0, w1, u0_temp, u1_temp, u0, u1, delta_temp;
	double delta_max = 0.0;
	for (int i = 0; i < 256; i++)
	{
		//��ʼ����ز���
		w0 = w1 = u0_temp = u1_temp = u0 = u1 = delta_temp = 0;
		for (int j = 0; j < 256; j++)
		{
			//��������
			if (j <= i)
			{
				//��ǰiΪ�ָ���ֵ����һ���ܵĸ���
				w0 += nProDis[j];
				u0_temp += j*nProDis[j];
			}
			//ǰ������
			else
			{
				//��ǰiΪ�ָ���ֵ����һ���ܵĸ���
				w1 += nProDis[j];
				u1_temp += j*nProDis[j];
			}
		}
		//�ֱ��������ƽ���Ҷ�
		u0 = u0_temp / w0;
		u1 = u1_temp / w1;
		delta_temp = (float)(w0*w1*pow((u0 - u1), 2));
		//�����ҵ������䷽���µ���ֵ
		if (delta_temp > delta_max)
		{
			delta_max = delta_temp;
			threshold = i;
		}
	}

	cout << threshold << endl;
	//����������ͼ��
	Mat otsuResultImage = Mat::zeros(srcGray.rows, srcGray.cols, CV_8UC1);
	//���õõ�����ֵʵ�ֶ�ֵ������
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