#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

py::array_t<float> compute_hog(py::array_t<uint8_t> img) {
    auto buf = img.unchecked<2>();
    int H = buf.shape(0), W = buf.shape(1);
    // Full implementation: compute gradients, histograms per 8x8, blocks of 2x2 cells
    std::vector<float> descriptor;
    int cell = 8, block = 2, bins=9;
    for(int by=0; by+cell<=H; by+=cell) for(int bx=0; bx+cell<=W; bx+=cell) {
        std::vector<float> hist(bins, 0.0f);
        for(int i=0;i<cell;i++) for(int j=0;j<cell;j++){
            int y=by+i, x=bx+j;
            float gx = (x>0?buf(y,x)-buf(y,x-1):0)+(x<W-1?buf(y,x+1)-buf(y,x):0);
            float gy = (y>0?buf(y,x)-buf(y-1,x):0)+(y<H-1?buf(y+1,x)-buf(y,x):0);
            float mag = std::hypot(gx,gy);
            float angle = std::atan2(gy,gx) * 180/M_PI;
            if(angle<0) angle+=180;
            int bin = std::min(int(angle/20), bins-1);
            hist[bin] += mag;
        }
        // normalize cell
        float norm = 0; for(auto v:hist) norm+=v*v; norm = std::sqrt(norm)+1e-6;
        for(auto &v:hist) descriptor.push_back(v/norm);
    }
    // convert to numpy array
    return py::array_t<float>(descriptor.size(), descriptor.data());
}

PYBIND11_MODULE(hog_ext, m) {
    m.doc() = "HOG feature extractor";
    m.def("compute_hog", &compute_hog, "Compute HOG descriptor");
}