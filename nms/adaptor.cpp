#include "../nms/include/pybind11/numpy.h"
#include "../nms/include/pybind11/pybind11.h"
#include "../nms/include/pybind11/stl.h"
#include "../nms/include/pybind11/stl_bind.h"
#include "../nms/nms.h"

namespace py = pybind11;

namespace cl = ClipperLib;

namespace nms_adaptor {

	std::vector<std::vector<float>> polys2floats(std::vector<nms::Polygon> &polys) {
		std::vector<std::vector<float>> ret;
		for (size_t i = 0; i < polys.size(); i ++) {
			auto &p = polys[i];
			auto &poly = p.poly;

			ret.emplace_back(std::vector<float>{
					float(poly[0].X), float(poly[0].Y),
					float(poly[1].X), float(poly[1].Y),
					float(poly[2].X), float(poly[2].Y),
					float(poly[3].X), float(poly[3].Y),
					float(p.score), float(p.nr_polys)
					});
		}

		return ret;
	}

	/**
	 *
	 * \param quad_n9 an n-by-9 numpy array, where first 8 numbers denote the
	 *		quadrangle, and the last one is the score
	 * \param iou_threshold two quadrangles with iou score above this threshold
	 *		will be merged
	 *
	 * \return an n-by-9 numpy array, the merged quadrangles
	 */
	std::vector<std::vector<float>> do_nms(
			py::array_t<float, py::array::c_style | py::array::forcecast> segm,
			py::array_t<float, py::array::c_style | py::array::forcecast> geo_map,
			py::array_t<float, py::array::c_style | py::array::forcecast> angle,
			py::array_t<int, py::array::c_style | py::array::forcecast> poly_map,
			float iou_threshold, float iou_threshold2, float segm_threshold) {
		auto ibuf = segm.request();
		auto pbuf = geo_map.request();
		auto abuf = angle.request();
		auto poly_buff = poly_map.request();
		if (pbuf.ndim != 3)
			throw std::runtime_error("geometry map must have a shape of (h x w x 4)");
		if (ibuf.ndim != 2)
				throw std::runtime_error("segmentation have a shape of (h x w)");
		if (abuf.ndim != 3)
				throw std::runtime_error("angle have a shape of (h x w x 2)");
		if (poly_buff.ndim != 2)
				throw std::runtime_error("polygon buffer have a shape of (h x w)");

		//TODO we are missing a lot of asserts ...

		int w = ibuf.shape[1];
		int h = ibuf.shape[0];
		int offset = 0;
		int rstride = w * 4;
		int astride = w * 2;
		float* iptr =  static_cast<float *>(ibuf.ptr);
		float* rptr =  static_cast<float *>(pbuf.ptr);
		float* aptr =  static_cast<float *>(abuf.ptr);
		int* poly_ptr = static_cast<int *>(poly_buff.ptr);
		float scale_factor = 4;

		float precision = 10000;

		std::vector<nms::Polygon> polys;
		using cInt = cl::cInt;
		for(int y = 0; y < h; y++){
			for(int x  = 0; x < w; x++){
				auto p = iptr + offset;
				auto r = rptr + y * rstride + x * 4;
				auto a = aptr + y * astride + x * 2;
				if( *p > segm_threshold ){
					float angle_cos = a[1];
				  float angle_sin = a[0];

				  float xp = x + 0.25f;
				  float yp = y + 0.25f;

				  float pos_r_x = (xp - r[2] * angle_cos) * scale_factor;
					float pos_r_y =	(yp - r[2] * angle_sin) * scale_factor;
					float pos_r2_x = (xp + r[3] * angle_cos) * scale_factor;
					float pos_r2_y = (yp + r[3] * angle_sin) * scale_factor;

				  float ph = 9;// (r[0] + r[1]) + 1e-5;
				  float phx = 9;

				  float p_left = expf(-r[2] / phx);
				  float p_top = expf(-r[0] / ph);
				  float p_right = expf(-r[3] / phx);
				  float p_bt = expf(-r[1] / ph);

				  nms::Polygon poly{
									{
										{cInt(roundf(precision * (pos_r_x  - r[1] * angle_sin * scale_factor))), cInt(roundf(precision * (pos_r_y + r[1] * angle_cos * scale_factor)))},
										{cInt(roundf(precision * (pos_r_x + r[0] * angle_sin * scale_factor ))), cInt(roundf(precision * (pos_r_y - r[0] * angle_cos * scale_factor)))},
										{cInt(roundf(precision * (pos_r2_x + r[0] * angle_sin * scale_factor))), cInt(roundf(precision * (pos_r2_y - r[0] * angle_cos * scale_factor)))},
										{cInt(roundf(precision * (pos_r2_x - r[1] * angle_sin * scale_factor))), cInt(roundf(precision * (pos_r2_y + r[1] * angle_cos * scale_factor)))},
									},
									p[0],
									{p_left * p_bt, p_left * p_top, p_right * p_top, p_right * p_bt},
									x,
									y,
									1
					};
					polys.push_back(poly);
				}
				offset++;
			}
		}
		std::vector<nms::Polygon> poly_out = nms::merge_iou(polys, poly_ptr, w, h, iou_threshold, iou_threshold2);
		return polys2floats(poly_out);
	}

}

PYBIND11_MODULE(adaptor, m) {

	m.def("do_nms", &nms_adaptor::do_nms,
				"perform non-maxima suppression");
}

