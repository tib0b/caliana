// Differential-testing oracle: drives the *original* pystackreg C++ TurboReg
// core (no Python / numpy) so its output can be compared against the Rust port.
//
// Usage:
//   oracle register  <w> <h> <transformation> <ref.bin> <mov.bin> <out_mat.bin>
//   oracle transform <w> <h> <ncols>          <mov.bin> <mat.bin> <out_img.bin>
//
// All image/matrix files are raw little-endian f64. Images are row-major with
// `w` columns and `h` rows (element (row,col) at row*w + col), matching the Rust
// side. `transformation` is the TurboReg code (2/3/4/6/8); for `transform`,
// `ncols` is the short-matrix column count (1/3/4).

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "TurboReg.h"
#include "TurboRegImage.h"
#include "TurboRegMask.h"
#include "TurboRegPointHandler.h"
#include "TurboRegTransform.h"
#include "matrix.h"

struct regMat {
  matrix<double> mat;
  matrix<double> refPts;
  matrix<double> movPts;
};

static std::vector<double> read_f64(const std::string &path, size_t n) {
  std::vector<double> v(n);
  std::ifstream is(path, std::ios::binary);
  is.read(reinterpret_cast<char *>(v.data()), n * sizeof(double));
  return v;
}

static void write_f64(const std::string &path, const double *p, size_t n) {
  std::ofstream os(path, std::ios::binary);
  os.write(reinterpret_cast<const char *>(p), n * sizeof(double));
}

static void registerImg(double *pDataRef, double *pDataMov,
                        Transformation transformation, int width, int height,
                        regMat &rm) {
  TurboRegImage refImg(pDataRef, width, height, transformation, true);
  TurboRegImage movImg(pDataMov, width, height, transformation, false);
  TurboRegPointHandler refPH(refImg, transformation);
  TurboRegPointHandler movPH(movImg, transformation);
  TurboRegMask refMsk(refImg);
  TurboRegMask movMsk(movImg);
  refMsk.clearMask();
  movMsk.clearMask();
  int pyramidDepth = getPyramidDepth(movImg.getWidth(), movImg.getHeight(),
                                     refImg.getWidth(), refImg.getHeight());
  refImg.setPyramidDepth(pyramidDepth);
  refMsk.setPyramidDepth(pyramidDepth);
  movImg.setPyramidDepth(pyramidDepth);
  movMsk.setPyramidDepth(pyramidDepth);
  refImg.init();
  refMsk.init();
  movImg.init();
  movMsk.init();
  TurboRegTransform tform(&movImg, &movMsk, &movPH, &refImg, &refMsk, &refPH,
                          transformation, false);
  tform.doRegistration();
  rm.mat = tform.getTransformationMatrix();
  rm.refPts = refPH.getPoints();
  rm.movPts = movPH.getPoints();
}

static std::vector<double> transformImg(matrix<double> m, double *pDataMov,
                                        int width, int height) {
  Transformation transformation = getTransformationFromMatrix(m);
  TurboRegImage movImg(pDataMov, width, height, transformation, false);
  TurboRegPointHandler movPH(movImg, transformation);
  TurboRegMask movMsk(movImg);
  movMsk.clearMask();
  int pyramidDepth = getPyramidDepth(movImg.getWidth(), movImg.getHeight(),
                                     movImg.getWidth(), movImg.getHeight());
  movImg.setPyramidDepth(pyramidDepth);
  movMsk.setPyramidDepth(pyramidDepth);
  movImg.init();
  movMsk.init();
  TurboRegTransform tform(&movImg, &movMsk, &movPH, transformation, false);
  return tform.doFinalTransform(&movImg, m);
}

int main(int argc, char **argv) {
  if (argc < 8) {
    fprintf(stderr, "bad args\n");
    return 2;
  }
  std::string mode = argv[1];
  int width = atoi(argv[2]);
  int height = atoi(argv[3]);

  if (mode == "register") {
    int tf = atoi(argv[4]);
    auto ref = read_f64(argv[5], (size_t)width * height);
    auto mov = read_f64(argv[6], (size_t)width * height);
    regMat rm;
    registerImg(ref.data(), mov.data(), (Transformation)tf, width, height, rm);
    write_f64(argv[7], rm.mat.begin(),
              (size_t)rm.mat.nrows() * rm.mat.ncols());
    return 0;
  } else if (mode == "transform") {
    int ncols = atoi(argv[4]);
    auto mov = read_f64(argv[5], (size_t)width * height);
    auto matdata = read_f64(argv[6], (size_t)2 * ncols);
    matrix<double> m(2, ncols);
    for (int i = 0; i < 2 * ncols; i++)
      m.begin()[i] = matdata[i];
    auto out = transformImg(m, mov.data(), width, height);
    write_f64(argv[7], out.data(), out.size());
    return 0;
  }
  fprintf(stderr, "unknown mode\n");
  return 2;
}
