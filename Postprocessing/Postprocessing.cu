#include "Postprocessing.cuh"
#include <algorithm>


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

int* gpu_maskSobel_x;
int* gpu_maskSobel_y;

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


__global__ void calculate_sobel2_gpu(int _width, uchar3* _srcData, uchar3* _outData, uchar3 _min, uchar3 _max)
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

	if ((newPixel.x + newPixel.y + newPixel.z)/3 < 160)
	{
		newPixel = { 0,0,0 };
	}
	unsigned char brightness = (newPixel.x + newPixel.y + newPixel.z) / 3;
	_outData[yPos * WIDTH + xPos] = { brightness  ,brightness ,brightness };
}



std::vector<std::vector<std::function<void(uchar3* _targetData, uchar3* _desData)>>> Postprocessing::postprocessingFunc;

void Postprocessing::init()
{
	cudaError_t Status = cudaSetDevice(0);

	//배열 메모리 할당
	pixelBuffer = new uchar3[WIDTH * HEIGHT];

	 Status = cudaMalloc((void**)&gpu_maskSobel_x, sizeof(int) * 9);
	assert(Status == cudaSuccess);
	Status = cudaMalloc((void**)&gpu_maskSobel_y, sizeof(int) * 9);
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
			calculate_sobel2_gpu<<<gridDim, blockDim >>>(WIDTH, gpu_tempBuffer1, gpu_outBuffer, gradientMin, gradientMax);
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
					if (brightness < 160)
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
			for (int y = 1; y < HEIGHT - 1; y++)
			{
				for (int x = 1; x < WIDTH - 1; x++)
				{
					_desData[y * WIDTH + x] = _targetData[y * WIDTH + x];
				}
			}
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
	

}
