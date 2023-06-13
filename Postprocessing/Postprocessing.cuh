#include <functional>
#include <vector>
#include "..\UniversalHookX\src\ThirdParty\glm\glm.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#define WIDTH 640
//#define HEIGHT 480

#define WIDTH 1024
#define HEIGHT 768


struct PixelInfo {
public:
	unsigned char r = 0;
	unsigned char g = 0;
	unsigned char b = 0;

	PixelInfo():r(0),g(0),b(0)
	{

	}

	PixelInfo(glm::vec3 _rgb) : r(_rgb.r),
		g(_rgb.g),
		b(_rgb.b)
	{}

	int get_brightness()
	{
		return (r + g + b) / 3;
	}

	unsigned char& operator[](int _idx)
	{
		assert(_idx >= 0 && _idx <= 2); // 오버 참조
		switch (_idx)
		{
		case 0:
			return r;
			break;
		case 1:
			return g;
			break;
		case 2:
			return b;
			break;
		default:
			return b;
			break;
		}
	}

};

enum class OperationMode
{
	CPU,
	GPU,
	Max
};

enum class Filter
{
	None,
	Contrast,
	Saturation,
	Brightness,
	Gray,
	Sobel,
	Gaussian_Blur,
	Sharpness,
	Max
};

class Postprocessing
{
public:
	Postprocessing();
	~Postprocessing();

	static void init();
	static void release();
	static void set_postprocessing(uchar3* _targetData, uchar3* _desData, OperationMode _mode, Filter _filter);

	static std::vector<std::vector<std::function<void(uchar3* _targetData, uchar3* _desData)>>> postprocessingFunc;
	static std::vector<std::vector<bool>> isEnable_postprocessing;


	// ----- Option ------
	static int sobel_threshold;
	static int gaussian_blur_threshold;
	static float sharpness_threshold;
	static float contrast_threshold;
	static float saturation_threshold;
	static float brightness_threshold;

};

