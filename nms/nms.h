#pragma once

#include "../nms/include/clipper/clipper.hpp"

namespace nms {

	namespace cl = ClipperLib;

	struct Polygon {
		cl::Path poly;
		float score;
		float probs[4];
		int x;
		int y;
		std::int32_t nr_polys;
	};

	float paths_area(const ClipperLib::Paths &ps) {
		float area = 0;
		for (auto &&p: ps)
			area += cl::Area(p);
		return area;
	}

	float poly_iou(const Polygon &a, const Polygon &b) {
		cl::Clipper clpr;
		clpr.AddPath(a.poly, cl::ptSubject, true);
		clpr.AddPath(b.poly, cl::ptClip, true);

		cl::Paths inter, uni;
		clpr.Execute(cl::ctIntersection, inter, cl::pftEvenOdd);
		clpr.Execute(cl::ctUnion, uni, cl::pftEvenOdd);

		auto inter_area = paths_area(inter),
			 uni_area = paths_area(uni);
		return std::abs(inter_area) / std::max(std::abs(uni_area), 1.0f);
	}

	bool should_merge(const Polygon &a, const Polygon &b, float iou_threshold) {
		return poly_iou(a, b) > iou_threshold;
	}

	/**
	 * Incrementally merge polygons
	 */
	class PolyMerger {
		public:
			PolyMerger(): score(0), nr_polys(0) {
				memset(data, 0, sizeof(data));
				memset(probs, 0, 4 * sizeof(float));
			}

			/**
			 * Add a new polygon to be merged.
			 */
			void add(const Polygon &p) {

				auto &poly = p.poly;
				data[0] += poly[0].X * p.probs[0];
				data[1] += poly[0].Y * p.probs[3];

				data[2] += poly[1].X * p.probs[0];
				data[3] += poly[1].Y * p.probs[1];

				data[4] += poly[2].X * p.probs[2];
				data[5] += poly[2].Y * p.probs[1];

				data[6] += poly[3].X * p.probs[2];
				data[7] += poly[3].Y * p.probs[3];

				score += p.score;

				probs[0] += p.probs[0];
				probs[1] += p.probs[1];
				probs[2] += p.probs[2];
				probs[3] += p.probs[3];

				nr_polys += p.nr_polys;
			}

			Polygon get() const {
				Polygon p;

				auto &poly = p.poly;
				poly.resize(4);

				poly[0].X = data[0] / probs[0];
				poly[0].Y = data[1] / probs[3];
				poly[1].X = data[2] / probs[0];
				poly[1].Y = data[3] / probs[1];
				poly[2].X = data[4] / probs[2];
				poly[2].Y = data[5] / probs[1];
				poly[3].X = data[6] / probs[2];
				poly[3].Y = data[7] / probs[3];

				assert(score > 0);
				p.score = score;
				p.probs[0] = probs[0];
				p.probs[1] = probs[1];
				p.probs[2] = probs[2];
				p.probs[3] = probs[3];
				p.nr_polys = nr_polys;

				return p;
			}

		private:
			std::int64_t data[8];
			float score;
			float probs[4];
			std::int32_t nr_polys;
	};


	/**
	 * The standard NMS algorithm.
	 */
	std::vector<Polygon> standard_nms(std::vector<Polygon> &polys, float iou_threshold) {
		size_t n = polys.size();
		if (n == 0)
			return {};
		std::vector<size_t> indices(n);
		std::iota(std::begin(indices), std::end(indices), 0);
		std::sort(std::begin(indices), std::end(indices), [&](size_t i, size_t j) { return polys[i].score > polys[j].score; });

		std::vector<size_t> keep;
		while (indices.size()) {
			size_t p = 0, cur = indices[0];
			keep.emplace_back(cur);
			for (size_t i = 1; i < indices.size(); i ++) {
				if (!should_merge(polys[cur], polys[indices[i]], iou_threshold)) {
					indices[p++] = indices[i];
				}else{
					PolyMerger merger;
					merger.add(polys[ indices[i]]);
					merger.add(polys[cur]);
					polys[cur] = merger.get();
				}
			}
			indices.resize(p);
		}

		std::vector<Polygon> ret;
		for (auto &&i: keep) {
			ret.emplace_back(polys[i]);
		}
		return ret;
	}


	std::vector<Polygon>
			merge_iou(std::vector<Polygon>& polys_in, int* poly_ptr, int w, int h, float iou_threshold1, float iou_threshold2) {

				// first pass
				std::vector<Polygon> polys;
				for (size_t i = 0; i < polys_in.size(); i ++) {
					auto poly = polys_in[i];

					if (polys.size()) {
						// merge with the last one
						auto &bpoly = polys.back();
						if (should_merge(poly, bpoly, iou_threshold1)) {
							PolyMerger merger;
							merger.add(bpoly);
							merger.add(poly);
							bpoly = merger.get();
							poly_ptr[poly.y * w + poly.x] = (polys.size() - 1);
							continue;
						}else{
							if(poly.y > 0){
								int idx = poly_ptr[(poly.y -1)* w + poly.x];
								if(idx >= 0){
									auto &cpoly = polys[idx];
									if (should_merge(poly, cpoly, iou_threshold1)) {
										PolyMerger merger;
										merger.add(cpoly);
										merger.add(poly);
										cpoly = merger.get();
										poly_ptr[poly.y * w + poly.x] = idx;
										continue;
									}
									if(poly.x > 0){
										idx = poly_ptr[(poly.y -1)* w + poly.x - 1];
										if(idx >= 0){
											auto &cpoly = polys[idx];
											if (should_merge(poly, cpoly, iou_threshold1)) {
												PolyMerger merger;
												merger.add(cpoly);
												merger.add(poly);
												cpoly = merger.get();
												poly_ptr[poly.y * w + poly.x] = idx;
												continue;
											}
										}
									}
									idx = poly_ptr[(poly.y -1)* w + poly.x + 1];
									if(idx >= 0){
										auto &cpoly = polys[idx];
										if (should_merge(poly, cpoly, iou_threshold1)) {
											PolyMerger merger;
											merger.add(cpoly);
											merger.add(poly);
											cpoly = merger.get();
											poly_ptr[poly.y * w + poly.x] = idx;
											continue;
										}
									}
								}
							}
							polys.emplace_back(poly);
						}
					}
					polys.emplace_back(poly);
					poly_ptr[poly.y * w + poly.x] = (polys.size() - 1);
				}
				return standard_nms(polys, iou_threshold2);
			}
}
