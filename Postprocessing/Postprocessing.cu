#include "Postprocessing.cuh"
#include <algorithm>
#include <cmath>

int MaskSobelX[9] = {
	-1, 0, 1,
	-2, 0, 2,
	-1, 0, 1
};

int MaskSobelY[9] = {
	1, 2, 1,
	0, 0, 0,
	-1, -2, -1
};

int mask_gaussianBlur[25] = {
	1, 4, 6, 4, 1,
	4,16,24,16, 4,
	6,24,36,24, 6,
	4,16,24,16, 4,
	1, 4, 6, 4, 1
};

int sharpness_mask[25] = {
	0,  0,  0,  0,  0,
	0,  0, -1,  0,  0,
	0, -1,  5, -1,  0,
	0,  0, -1,  0,  0,
	0,  0,  0,  0,  0,
};


int* gpu_maskSobel_x;
int* gpu_maskSobel_y;
int* gpu_gaussianBlur_mask;
int* sharpness_mask_gpu;

uchar3* pixelBuffer;

uchar3* gpu_srcBuffer;
uchar3* gpu_outBuffer;

uchar3* gpu_tempBuffer1;

__global__ void copy_image(int _width, uchar3* _srcData, uchar3* _outData)
{
	//좌표를 파악
	int xPos = blockIdx.x * blockDim.x + threadIdx.x;
	int yPos = blockIdx.y * blockDim.y + threadIdx.y;


	_outData[yPos * _width + xPos] = _srcData[yPos * _width + xPos];
}

__global__ void gpu_gray(int _width, uchar3* _srcData, uchar3* _outData)
{
	//좌표를 파악
	int xPos = blockIdx.x * blockDim.x + threadIdx.x;
	int yPos = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned char brightness = (_srcData[yPos * _width + xPos].x + _srcData[yPos * _width + xPos].y + _srcData[yPos * _width + xPos].z) / 3;
	_outData[yPos * _width + xPos] = { brightness ,brightness ,brightness };
}

__global__ void contrast_gpu(int _width, uchar3* _srcData, uchar3* _outData, float _strength)
{
	//좌표를 파악
	int xPos = blockIdx.x * blockDim.x + threadIdx.x;
	int yPos = blockIdx.y * blockDim.y + threadIdx.y;

	uchar3 pixelColor = _srcData[yPos * WIDTH + xPos];

	pixelColor.x = (static_cast<float>(pixelColor.x - 127) * _strength) + 127;
	pixelColor.y = (static_cast<float>(pixelColor.y - 127) * _strength) + 127;
	pixelColor.z = (static_cast<float>(pixelColor.z - 127) * _strength) + 127;

	_outData[yPos * _width + xPos] = pixelColor;
}

__global__ void saturation_gpu(int _width, uchar3* _srcData, uchar3* _outData, float _strength)
{
	//좌표를 파악
	int xPos = blockIdx.x * blockDim.x + threadIdx.x;
	int yPos = blockIdx.y * blockDim.y + threadIdx.y;

	uchar3 pixelColor = _srcData[yPos * WIDTH + xPos];
	unsigned char avg = (pixelColor.x + pixelColor.y + pixelColor.z) / 3;

	pixelColor.x = avg * (1.0f - _strength) + pixelColor.x * _strength;
	pixelColor.y = avg * (1.0f - _strength) + pixelColor.y * _strength;
	pixelColor.z = avg * (1.0f - _strength) + pixelColor.z * _strength;

	_outData[yPos * _width + xPos] = pixelColor;
}

__global__ void brightness_gpu(int _width, uchar3* _srcData, uchar3* _outData, float _strength)
{
	//좌표를 파악
	int xPos = blockIdx.x * blockDim.x + threadIdx.x;
	int yPos = blockIdx.y * blockDim.y + threadIdx.y;

	uchar3 pixelColor = _srcData[yPos * WIDTH + xPos];


	pixelColor.x = 0.5f * ((float)pixelColor.x *_strength) + 0.5f * (float)pixelColor.x;
	pixelColor.y = 0.5f * ((float)pixelColor.y *_strength) + 0.5f * (float)pixelColor.y;
	pixelColor.z = 0.5f * ((float)pixelColor.z *_strength) + 0.5f * (float)pixelColor.z;


	_outData[yPos * _width + xPos] = pixelColor;
}

__global__ void apply_mask_gpu(int _width, uchar3* _srcData, uchar3* _outData ,int* _mask,int _maskSize, int _maskWidth,int _threshold)
{
	int xPos = blockIdx.x * blockDim.x + threadIdx.x;
	int yPos = blockIdx.y * blockDim.y + threadIdx.y;

	float3 convolution = { 0,0,0 };
	//현재 픽셀의 위치 (xPos,yPos)
	for (int k = 0; k < _maskSize; k++)
	{
		int r = k / _maskWidth; 
		int c = k % _maskWidth;

		int idx = (yPos + r - 1) * _width + (xPos + c - 1);
		convolution.x = convolution.x + _mask[k] * _srcData[idx].x / (_threshold);
		if (convolution.x > 255)
			convolution.x = 255;
		else if (convolution.x < 0)
			convolution.x = 0;

		convolution.y = convolution.y +  _mask[k] * _srcData[idx].y / (_threshold);
		if (convolution.y > 255)
			convolution.y = 255;
		else if (convolution.y < 0)
			convolution.y = 0;

		convolution.z = convolution.z +  _mask[k] * _srcData[idx].z / (_threshold);
		if (convolution.z > 255)
			convolution.z = 255;
		else if (convolution.z < 0)
			convolution.z = 0;


	}
	_outData[yPos * WIDTH + xPos] = {static_cast<unsigned char>(convolution.x),static_cast<unsigned char>(convolution.y),static_cast<unsigned char>(convolution.z)};
}

__global__ void calculate_sobel_gpu(int _width, uchar3* _srcData, uchar3* _outData, int* _maskSobelX, int* _maskSobelY)
{
	//좌표를 파악
	int xPos = blockIdx.x * blockDim.x + threadIdx.x;
	int yPos = blockIdx.y * blockDim.y + threadIdx.y;

	uchar3 gradientX = {0,0,0};
	uchar3 gradientY = { 0,0,0 };
	//현재 픽셀의 위치 (xPos,yPos)
	for (int k = 0; k < 9; k++)
	{
		int r = k / 3; //k = 0,1,2 때 r = 0 , k = 3,4,5 때 r =1
		int c = k % 3;
		// k = 0 일 때 i,j = (x - 1, y - 1)
		// k = 1 일 때 i,j = (x, y - 1)
		// k = 0 일 때 (r,c ) = (0,0) (r-1, c - 1) = (-1, -1)
		// k = 1 일 때 (r, c) = (0,1) (r-1, c-1)	= (-1,0)
		// k = 2 일 때 (r, c) = (0, 2) (r-1, c-1) = (-1, 1)
		int idx = (yPos + r - 1) * _width + (xPos + c - 1);
		gradientX.x = gradientX.x + _maskSobelX[k] * _srcData[idx].x;
		gradientY.x = gradientY.x + _maskSobelY[k] * _srcData[idx].x;

		gradientX.y = gradientX.y + _maskSobelX[k] * _srcData[idx].y;
		gradientY.y = gradientY.y + _maskSobelY[k] * _srcData[idx].y;

		gradientX.z = gradientX.z + _maskSobelX[k] * _srcData[idx].z;
		gradientY.z = gradientY.z + _maskSobelY[k] * _srcData[idx].z;
	}
	uchar3 magnitude;
	magnitude.x = sqrtf(gradientX.x * gradientX.x + gradientY.x * gradientY.x);
	magnitude.y = sqrtf(gradientX.y * gradientX.y + gradientY.y * gradientY.y);
	magnitude.z = sqrtf(gradientX.z * gradientX.z + gradientY.z * gradientY.z);

	_outData[yPos * WIDTH + xPos] = magnitude;
}


__global__ void calculate_sobel2_gpu(int _width, uchar3* _srcData, uchar3* _outData, uchar3 _min, uchar3 _max, int _threshold)
{
	//좌표를 파악
	int xPos = blockIdx.x * blockDim.x + threadIdx.x;
	int yPos = blockIdx.y * blockDim.y + threadIdx.y;


	int currentIndex = _width * yPos + xPos;

	uchar3 newPixel;
	{
		float g = _srcData[yPos * _width + xPos].x;
		float t = (g - _min.x) / (_max.x - _min.x);
		newPixel.x = static_cast<unsigned char>(t * 255);
	}
	{
		float g = _srcData[yPos * _width + xPos].y;
		float t = (g - _min.y) / (_max.y - _min.y);
		newPixel.y = static_cast<unsigned char>(t * 255);
	}
	{
		float g = _srcData[yPos * _width + xPos].z;
		float t = (g - _min.z) / (_max.z - _min.z);
		newPixel.z = static_cast<unsigned char>(t * 255);
	}

	if ((newPixel.x + newPixel.y + newPixel.z)/3 < _threshold)
	{
		newPixel = { 0,0,0 };
	}
	unsigned char brightness = (newPixel.x + newPixel.y + newPixel.z) / 3;
	_outData[yPos * WIDTH + xPos] = { brightness  ,brightness ,brightness };
}

void copy_data_cpu(uchar3* _des,uchar3* _target)
{
	for (int y = 1; y < HEIGHT - 1; y++)
	{
		for (int x = 1; x < WIDTH - 1; x++)
		{
			_des[y * WIDTH + x] = _target[y * WIDTH + x];
		}
	}
}


std::vector<std::vector<std::function<void(uchar3* _targetData, uchar3* _desData)>>> Postprocessing::postprocessingFunc;
std::vector<std::vector<bool>> Postprocessing::isEnable_postprocessing;

int Postprocessing::sobel_threshold = 160;
int Postprocessing::gaussian_blur_threshold = 256;
float Postprocessing::sharpness_threshold = 1.f;
float Postprocessing::contrast_threshold =   0.5f;
float Postprocessing::saturation_threshold = 0.5f;
float Postprocessing::brightness_threshold = 0.5f;
void Postprocessing::init()
{
	cudaError_t Status = cudaSetDevice(0);

	//배열 메모리 할당
	pixelBuffer = new uchar3[WIDTH * HEIGHT];

	 Status = cudaMalloc((void**)&gpu_maskSobel_x, sizeof(int) * 9);
	assert(Status == cudaSuccess);
	Status = cudaMalloc((void**)&gpu_maskSobel_y, sizeof(int) * 9);
	assert(Status == cudaSuccess);
	Status = cudaMalloc((void**)&gpu_gaussianBlur_mask, sizeof(int) * 25);
	assert(Status == cudaSuccess);
	Status = cudaMalloc((void**)&sharpness_mask_gpu, sizeof(int) * 25);
	assert(Status == cudaSuccess);
	Status = cudaMalloc((void**)&gpu_srcBuffer, sizeof(uchar3) * WIDTH * HEIGHT);
	assert(Status == cudaSuccess);
	Status = cudaMalloc((void**)&gpu_outBuffer, sizeof(uchar3) * WIDTH * HEIGHT);
	assert(Status == cudaSuccess);
	Status = cudaMalloc((void**)&gpu_tempBuffer1, sizeof(uchar3) * WIDTH * HEIGHT);
	assert(Status == cudaSuccess);

	Status = cudaMemcpy(gpu_maskSobel_x, MaskSobelX, sizeof(int) * 9, cudaMemcpyHostToDevice);
	assert(Status == cudaSuccess);
	Status = cudaMemcpy(gpu_maskSobel_y, MaskSobelY, sizeof(int) * 9, cudaMemcpyHostToDevice);
	assert(Status == cudaSuccess);
	Status = cudaMemcpy(gpu_gaussianBlur_mask, mask_gaussianBlur, sizeof(int) * 25, cudaMemcpyHostToDevice);
	assert(Status == cudaSuccess);
	Status = cudaMemcpy(sharpness_mask_gpu, sharpness_mask, sizeof(int) * 25, cudaMemcpyHostToDevice);
	assert(Status == cudaSuccess);


	//포스트 프로세싱 Enable 초기화
	isEnable_postprocessing.resize(static_cast<int>(OperationMode::Max));
	for (int i = 0; i < isEnable_postprocessing.size(); i++)
	{
		isEnable_postprocessing[i].resize(static_cast<int>(OperationMode::Max));
		for (int j = 0; j < isEnable_postprocessing[i].size(); j++)
		{
			isEnable_postprocessing[i][j] = false;
		}

	}

	//포스트 프로세싱 함수 바인드
	postprocessingFunc.resize(static_cast<int>(OperationMode::Max));
	for (int i = 0; i < postprocessingFunc.size(); i++)
	{
		postprocessingFunc[i].resize(static_cast<int>(Filter::Max));
	}

	postprocessingFunc[static_cast<int>(OperationMode::GPU)][static_cast<int>(Filter::Sobel)] = std::bind([=](uchar3* _targetData, uchar3* _desData)
		{
			cudaError_t Status = cudaMemcpy(gpu_srcBuffer, _targetData, sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);
			assert(Status == cudaSuccess);
			dim3 gridDim(32, 24, 1);
			dim3 blockDim(32, 32, 1);
			calculate_sobel_gpu << <gridDim, blockDim >> > (WIDTH, gpu_srcBuffer, gpu_tempBuffer1, gpu_maskSobel_x, gpu_maskSobel_y);
			cudaDeviceSynchronize();

			Status = cudaMemcpy(_desData, gpu_tempBuffer1, sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
			assert(Status == cudaSuccess);

			uchar3 gradientMin = *std::min_element(_desData, _desData + WIDTH * HEIGHT - 1,[](uchar3 _a, uchar3 _b)
				{
					return (_a.x + _a.y + _a.z) < (_b.x + _b.y + _b.z);
				});
			uchar3 gradientMax = *std::max_element(_desData, _desData + WIDTH * HEIGHT - 1, [](uchar3 _a, uchar3 _b)
				{
					return (_a.x + _a.y + _a.z) < (_b.x + _b.y + _b.z);
				});
			calculate_sobel2_gpu<<<gridDim, blockDim >>>(WIDTH, gpu_tempBuffer1, gpu_outBuffer, gradientMin, gradientMax,sobel_threshold);
			cudaDeviceSynchronize();

			Status = cudaMemcpy(_desData, gpu_outBuffer, sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
			assert(Status == cudaSuccess);

		},
		std::placeholders::_1, std::placeholders::_2);

	postprocessingFunc[static_cast<int>(OperationMode::CPU)][static_cast<int>(Filter::Sobel)] = std::bind([=](uchar3* _targetData, uchar3* _desData)
		{
			//그래디언트 크기 구하기
			for (int y = 1; y < HEIGHT - 1; y++)
			{
				for (int x = 1; x < WIDTH - 1; x++)
				{
					uchar3 gradientX = {0,0,0};
					uchar3 gradientY = {0,0,0};
					//현재 픽셀의 위치 (x,y)
					for (int k = 0; k < 9; k++)
					{
						int r = k / 3; //k = 0,1,2 때 r = 0 , k = 3,4,5 때 r =1
						int c = k % 3;
						// k = 0 일 때 i,j = (x - 1, y - 1)
						// k = 1 일 때 i,j = (x, y - 1)
						// k = 0 일 때 (r,c ) = (0,0) (r-1, c - 1) = (-1, -1)
						// k = 1 일 때 (r, c) = (0,1) (r-1, c-1)	= (-1,0)
						// k = 2 일 때 (r, c) = (0, 2) (r-1, c-1) = (-1, 1)
						int idx = (y + r - 1) * WIDTH + (x + c - 1);


						gradientX.x = gradientX.x + MaskSobelX[k] * _targetData[idx].x;
						gradientY.x = gradientY.x + MaskSobelY[k] * _targetData[idx].x;

						gradientX.y = gradientX.y + MaskSobelX[k] * _targetData[idx].y;
						gradientY.y = gradientY.y + MaskSobelY[k] * _targetData[idx].y;

						gradientX.z = gradientX.z + MaskSobelX[k] * _targetData[idx].z;
						gradientY.z = gradientY.z + MaskSobelY[k] * _targetData[idx].z;
					}
					uchar3 magnitude;
					magnitude.x = sqrtf(gradientX.x * gradientX.x + gradientY.x * gradientY.x);
					magnitude.y = sqrtf(gradientX.y * gradientX.y + gradientY.y * gradientY.y);
					magnitude.z = sqrtf(gradientX.z * gradientX.z + gradientY.z * gradientY.z);
					//float magnitude = sqrtf(Gx * Gx + Gy * Gy);
					pixelBuffer[y * WIDTH + x] = magnitude;
				}
			}

			////결과 이미지 생성
			//for (int y = 1; y < HEIGHT - 1; y++)
			//{
			//	for (int x = 1; x < WIDTH - 1; x++)
			//	{
			//		_desData[y * WIDTH + x] = pixelBuffer[y * WIDTH + x];
			//	}
			//}

			//그래디언트 크기의 최대값과 최솟값 구하기
			uchar3 min = { 255,255,255 };
			uchar3 max = { 0,0,0 };

			for (int y = 1; y < HEIGHT - 1; y++)
			{
				for (int x = 1; x < WIDTH - 1; x++)
				{
					int idx = y * WIDTH + x;

					int buffer_brightness = (pixelBuffer[idx].x + pixelBuffer[idx].y + pixelBuffer[idx].z) / 3;
					int min_brightness = (min.x + min.y + min.z) / 3;
					int max_brightness = (max.x + max.y + max.z) / 3;

					if (buffer_brightness < min_brightness)
						min = pixelBuffer[idx];
					if (buffer_brightness > max_brightness)
						max = pixelBuffer[idx];
				}
			}

			//결과 이미지 생성
			for (int y = 1; y < HEIGHT - 1; y++)
			{
				for (int x = 1; x < WIDTH - 1; x++)
				{
					uchar3 newPixel;
					{
						float g = pixelBuffer[y * WIDTH + x].x;
						float t = (g - min.x) / (max.x - min.x);
						newPixel.x = static_cast< unsigned char>((t ) * 255);
					}
					{
						float g = pixelBuffer[y * WIDTH + x].y;
						float t = (g - min.y) / (max.y - min.y);
						newPixel.y = static_cast<unsigned char>((t ) * 255);
					}
					{
						float g = pixelBuffer[y * WIDTH + x].z;
						float t = (g - min.z) / (max.z - min.z);
						newPixel.z = static_cast<unsigned char>((t ) * 255);
					}

					int brightness = (newPixel.x + newPixel.y + newPixel.z) / 3;
					if (brightness < sobel_threshold)
					{
						brightness = 0;
					}

					_desData[y * WIDTH + x] = { (unsigned char)brightness,(unsigned char)brightness ,(unsigned char)brightness };
				}
			}
		},
		std::placeholders::_1, std::placeholders::_2);

	postprocessingFunc[static_cast<int>(OperationMode::CPU)][static_cast<int>(Filter::None)] = std::bind([=](uchar3* _targetData, uchar3* _desData)
		{
			//결과 이미지 생성
			copy_data_cpu(_desData, _targetData);
		}
	, std::placeholders::_1, std::placeholders::_2);

	postprocessingFunc[static_cast<int>(OperationMode::GPU)][static_cast<int>(Filter::None)] = std::bind([=](uchar3* _targetData, uchar3* _desData)
		{
			cudaError_t Status = cudaMemcpy(gpu_srcBuffer, _targetData, sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);
			assert(Status == cudaSuccess);

			dim3 gridDim(32,24,1);
			dim3 blockDim(32, 32, 1);
			copy_image << <gridDim, blockDim >> > (WIDTH, gpu_srcBuffer, gpu_outBuffer);
			cudaDeviceSynchronize();
			Status = cudaMemcpy(_desData, gpu_outBuffer, sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
			assert(Status == cudaSuccess);


		}
	, std::placeholders::_1, std::placeholders::_2);

	postprocessingFunc[static_cast<int>(OperationMode::CPU)][static_cast<int>(Filter::Gray)] = std::bind([=](uchar3* _targetData, uchar3* _desData)
		{
			//결과 이미지 생성
			for (int y = 1; y < HEIGHT - 1; y++)
			{
				for (int x = 1; x < WIDTH - 1; x++)
				{
					unsigned char brightness = (_targetData[y * WIDTH + x].x + _targetData[y * WIDTH + x].y + _targetData[y * WIDTH + x].z) / 3;
					_desData[y * WIDTH + x] = { brightness ,brightness ,brightness };
				}
			}
		}
	, std::placeholders::_1, std::placeholders::_2);

	postprocessingFunc[static_cast<int>(OperationMode::GPU)][static_cast<int>(Filter::Gray)] = std::bind([=](uchar3* _targetData, uchar3* _desData)
		{
			cudaError_t Status = cudaMemcpy(gpu_srcBuffer, _targetData, sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);
			assert(Status == cudaSuccess);

			dim3 gridDim(32, 24, 1);
			dim3 blockDim(32, 32, 1);
			gpu_gray << <gridDim, blockDim >> > (WIDTH, gpu_srcBuffer, gpu_outBuffer);
			cudaDeviceSynchronize();
			Status = cudaMemcpy(_desData, gpu_outBuffer, sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
			assert(Status == cudaSuccess);


		}
	, std::placeholders::_1, std::placeholders::_2);

	postprocessingFunc[static_cast<int>(OperationMode::CPU)][static_cast<int>(Filter::Gaussian_Blur)] = std::bind([=](uchar3* _targetData, uchar3* _desData)
		{
			for (int y = 1; y < HEIGHT - 1; y++)
			{
				for (int x = 1; x < WIDTH - 1; x++)
				{
					uchar3 convolution = { 0,0,0 };
					//현재 픽셀의 위치 (x,y)
					for (int k = 0; k < 25; k++)
					{
						int r = k / 5; 
						int c = k % 5;
						int idx = (y + r - 1) * WIDTH + (x + c - 1);

						
						convolution.x += mask_gaussianBlur[k] *(1.f/  (float)gaussian_blur_threshold) *_targetData[idx].x;

						convolution.y += mask_gaussianBlur[k] * (1.f/ (float)gaussian_blur_threshold) * _targetData[idx].y;

						convolution.z += mask_gaussianBlur[k] * (1.f/ (float)gaussian_blur_threshold) * _targetData[idx].z;
					}
					pixelBuffer[y * WIDTH + x] = convolution;
				}
			}

			copy_data_cpu(_desData, pixelBuffer);
		}
	, std::placeholders::_1, std::placeholders::_2);

	postprocessingFunc[static_cast<int>(OperationMode::GPU)][static_cast<int>(Filter::Gaussian_Blur)] = std::bind([=](uchar3* _targetData, uchar3* _desData)
		{
			cudaError_t Status = cudaMemcpy(gpu_srcBuffer, _targetData, sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);
			assert(Status == cudaSuccess);
			dim3 gridDim(32, 24, 1);
			dim3 blockDim(32, 32, 1);
			//int _width, uchar3* _srcData, uchar3* _outData ,int* _mask,int _maskSize, int _maskWidth,int _threshold
			apply_mask_gpu << <gridDim, blockDim >> > (WIDTH, gpu_srcBuffer, gpu_outBuffer,gpu_gaussianBlur_mask,25,5,gaussian_blur_threshold);
			cudaDeviceSynchronize();
			Status = cudaMemcpy(_desData, gpu_outBuffer, sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
			assert(Status == cudaSuccess);
		}
	, std::placeholders::_1, std::placeholders::_2);

	postprocessingFunc[static_cast<int>(OperationMode::CPU)][static_cast<int>(Filter::Contrast)] = 
		std::bind([=](uchar3* _targetData, uchar3* _desData)
		{
			//결과 이미지 생성
			for (int y = 1; y < HEIGHT - 1; y++)
			{
				for (int x = 1; x < WIDTH - 1; x++)
				{
					uchar3 pixelColor = _desData[y * WIDTH + x];
					
					pixelColor.x = (static_cast<float>(pixelColor.x - 127) * contrast_threshold) + 127;
					pixelColor.y = (static_cast<float>(pixelColor.y - 127) * contrast_threshold) + 127;
					pixelColor.z = (static_cast<float>(pixelColor.z - 127) * contrast_threshold) + 127;


					_desData[y * WIDTH + x] = pixelColor;
				}
			}
		}
	, std::placeholders::_1, std::placeholders::_2);

	postprocessingFunc[static_cast<int>(OperationMode::GPU)][static_cast<int>(Filter::Contrast)] = std::bind([=](uchar3* _targetData, uchar3* _desData)
		{
			//TargetData를 GPU 버퍼에 복사
			cudaError_t Status = cudaMemcpy(gpu_srcBuffer, _targetData, sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);
			assert(Status == cudaSuccess);

			dim3 gridDim(32, 24, 1); // 1024 * 768
			dim3 blockDim(32, 32, 1);
			contrast_gpu << <gridDim, blockDim >> > (WIDTH, gpu_srcBuffer,gpu_outBuffer,contrast_threshold); 
			cudaDeviceSynchronize();

			//연산 결과를 DesData에 복사
			Status = cudaMemcpy(_desData, gpu_outBuffer, sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
			assert(Status == cudaSuccess);
		}
	, std::placeholders::_1, std::placeholders::_2);

	postprocessingFunc[static_cast<int>(OperationMode::CPU)][static_cast<int>(Filter::Saturation)] = std::bind([=](uchar3* _targetData, uchar3* _desData)
		{
			//결과 이미지 생성
			for (int y = 1; y < HEIGHT - 1; y++)
			{
				for (int x = 1; x < WIDTH - 1; x++)
				{
					uchar3 pixelColor = _desData[y * WIDTH + x];
					unsigned char avg = (pixelColor.x + pixelColor.y + pixelColor.z) / 3;

					pixelColor.x = avg * (1.0f - saturation_threshold) + pixelColor.x * saturation_threshold;
					pixelColor.y = avg * (1.0f - saturation_threshold) + pixelColor.y * saturation_threshold;
					pixelColor.z = avg * (1.0f - saturation_threshold) + pixelColor.z * saturation_threshold;

					_desData[y * WIDTH + x] = pixelColor;
				}
			}
		}
	, std::placeholders::_1, std::placeholders::_2);

	postprocessingFunc[static_cast<int>(OperationMode::GPU)][static_cast<int>(Filter::Saturation)] = std::bind([=](uchar3* _targetData, uchar3* _desData)
		{
			cudaError_t Status = cudaMemcpy(gpu_srcBuffer, _targetData, sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);
			assert(Status == cudaSuccess);

			dim3 gridDim(32, 24, 1);
			dim3 blockDim(32, 32, 1);
			saturation_gpu << <gridDim, blockDim >> > (WIDTH, gpu_srcBuffer, gpu_outBuffer, saturation_threshold);
			cudaDeviceSynchronize();
			Status = cudaMemcpy(_desData, gpu_outBuffer, sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
			assert(Status == cudaSuccess);
		}
	, std::placeholders::_1, std::placeholders::_2);

	postprocessingFunc[static_cast<int>(OperationMode::CPU)][static_cast<int>(Filter::Brightness)] = std::bind([=](uchar3* _targetData, uchar3* _desData)
		{
			//결과 이미지 생성
			for (int y = 1; y < HEIGHT - 1; y++)
			{
				for (int x = 1; x < WIDTH - 1; x++)
				{
					uchar3 pixelColor = _desData[y * WIDTH + x];

					pixelColor.x = 0.5f * ((float)pixelColor.x * brightness_threshold) + 0.5f *(float)pixelColor.x;
					pixelColor.y = 0.5f * ((float)pixelColor.y * brightness_threshold) + 0.5f *(float)pixelColor.y;
					pixelColor.z = 0.5f * ((float)pixelColor.z * brightness_threshold) + 0.5f *(float)pixelColor.z;

					_desData[y * WIDTH + x] = pixelColor;
				}
			}
		}
	, std::placeholders::_1, std::placeholders::_2);

	postprocessingFunc[static_cast<int>(OperationMode::GPU)][static_cast<int>(Filter::Brightness)] = std::bind([=](uchar3* _targetData, uchar3* _desData)
		{
			cudaError_t Status = cudaMemcpy(gpu_srcBuffer, _targetData, sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);
			assert(Status == cudaSuccess);

			dim3 gridDim(32, 24, 1);
			dim3 blockDim(32, 32, 1);
			brightness_gpu << <gridDim, blockDim >> > (WIDTH, gpu_srcBuffer, gpu_outBuffer, brightness_threshold);
			cudaDeviceSynchronize();
			Status = cudaMemcpy(_desData, gpu_outBuffer, sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
			assert(Status == cudaSuccess);
		}
	, std::placeholders::_1, std::placeholders::_2);

	postprocessingFunc[static_cast<int>(OperationMode::CPU)][static_cast<int>(Filter::Sharpness)] = std::bind([=](uchar3* _targetData, uchar3* _desData)
		{
			for (int y = 1; y < HEIGHT - 1; y++)
			{
				for (int x = 1; x < WIDTH - 1; x++)
				{
					uchar3 convolution = { 0,0,0 };
					//현재 픽셀의 위치 (x,y)
					for (int k = 0; k < 25; k++)
					{
						int r = k / 5;
						int c = k % 5;
						int idx = (y + r - 1) * WIDTH + (x + c - 1);


						convolution.x = convolution.x + (1.f / (float)sharpness_threshold) * sharpness_mask[k] * _targetData[idx].x;

						convolution.y = convolution.y + (1.f / (float)sharpness_threshold) * sharpness_mask[k] * _targetData[idx].y;

						convolution.z = convolution.z + (1.f / (float)sharpness_threshold) * sharpness_mask[k] * _targetData[idx].z;
					}
					pixelBuffer[y * WIDTH + x] = convolution;
				}
			}

			copy_data_cpu(_desData, pixelBuffer);
		}
	, std::placeholders::_1, std::placeholders::_2);

	postprocessingFunc[static_cast<int>(OperationMode::GPU)][static_cast<int>(Filter::Sharpness)] = std::bind([=](uchar3* _targetData, uchar3* _desData)
		{
			cudaError_t Status = cudaMemcpy(gpu_srcBuffer, _targetData, sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyHostToDevice);
			assert(Status == cudaSuccess);
			dim3 gridDim(32, 24, 1);
			dim3 blockDim(32, 32, 1);
			//int _width, uchar3* _srcData, uchar3* _outData ,int* _mask,int _maskSize, int _maskWidth,int _threshold
			apply_mask_gpu << <gridDim, blockDim >> > (WIDTH, gpu_srcBuffer, gpu_outBuffer, sharpness_mask_gpu, 25, 5, sharpness_threshold);
			cudaDeviceSynchronize();
			Status = cudaMemcpy(_desData, gpu_outBuffer, sizeof(uchar3) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
			assert(Status == cudaSuccess);
		}
	, std::placeholders::_1, std::placeholders::_2);
}

void Postprocessing::release()
{
	delete[] pixelBuffer;
}

void Postprocessing::set_postprocessing(uchar3* _targetData, uchar3* _desData, OperationMode _mode, Filter _filter)
{
	//버퍼 초기화
	memset(pixelBuffer, 0, sizeof(uchar3) * WIDTH * HEIGHT);
	postprocessingFunc[static_cast<int>(_mode)][static_cast<int>(_filter)](_targetData, _desData);

	//for (int mode = 0; mode < postprocessingFunc.size(); mode++)
	//{
	//	for (int filter = 0; filter < postprocessingFunc[mode].size(); filter++)
	//	{
	//		if ((isEnable_postprocessing[mode][filter]) == true)
	//		{
	//			memset(pixelBuffer, 0, sizeof(uchar3) * WIDTH * HEIGHT);
	//			


	//		}
	//	}
	//}
}
