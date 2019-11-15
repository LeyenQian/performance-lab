#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <pthread.h>
#include "cs1300bmp.h"
#include <iostream>
#include <fstream>
#include "Filter.h"

using namespace std;

#include "rdtsc.h"
extern "C"
{
//#include <emmintrin.h>
//#include <mmintrin.h>
#include <immintrin.h> // avx
//#include <smmintrin.h> // sse
}

//
// Forward declare the functions
//

Filter *readFilter(string filename);
double applyFilter(Filter *filter, cs1300bmp *input, cs1300bmp *output);


// for multi-threading
struct args {
    struct Filter *filter;
    cs1300bmp *input;
    cs1300bmp *output;
    int plane;
};
void *applyFilterPlaneSSE(void *input);
void *applyFilterNormal(void *arg);
void *applyFilterAVXwithMemcpy(void *arg);
void *applyFilterAVXwithDuffDevice(void *arg);

int main(int argc, char **argv)
{

  if (argc < 2)
  {
    fprintf(stderr, "Usage: %s filter inputfile1 inputfile2 .... \n", argv[0]);
  }

  //
  // Convert to C++ strings to simplify manipulation
  //
  string filtername = argv[1];

  //
  // remove any ".filter" in the filtername
  //
  string filterOutputName = filtername;
  string::size_type loc = filterOutputName.find(".filter");
  if (loc != string::npos)
  {
    //
    // Remove the ".filter" name, which should occur on all the provided filters
    //
    filterOutputName = filtername.substr(0, loc);
  }

  Filter *filter = readFilter(filtername);

  double sum = 0.0;
  int samples = 0;
  cout << endl << "**********************************************************************" << endl;
  cout << "Using AVX and memcpy for copying each row" << endl;
  for (int inNum = 2; inNum < argc; inNum++)
  {
    string inputFilename = argv[inNum];
    string outputFilename = "filtered-" + filterOutputName + "-" + inputFilename;
    struct cs1300bmp *input = new struct cs1300bmp;
    struct cs1300bmp *output = new struct cs1300bmp;

    posix_memalign(reinterpret_cast<void **>(&input), 32, sizeof(cs1300bmp));
    posix_memalign(reinterpret_cast<void **>(&output), 32, sizeof(cs1300bmp));

    int ok = cs1300bmp_readfile((char *)inputFilename.c_str(), input);

    if (ok)
    {
      pthread_t th0;
      struct args *th0Args = (struct args *)malloc(sizeof(args));
      th0Args->filter = filter;
      th0Args->input = input;
      th0Args->output = output;
      th0Args->plane = 0;

      pthread_t th1;
      struct args *th1Args = (struct args *)malloc(sizeof(args));
      th1Args->filter = filter;
      th1Args->input = input;
      th1Args->output = output;
      th1Args->plane = 1;

      pthread_t th2;
      struct args *th2Args = (struct args *)malloc(sizeof(args));
      th2Args->filter = filter;
      th2Args->input = input;
      th2Args->output = output;
      th2Args->plane = 2;

      long long cycStart, cycStop;
      cycStart = rdtscll();

      pthread_create(&th0, nullptr, applyFilterAVXwithMemcpy, (void *)th0Args);
      pthread_create(&th1, nullptr, applyFilterAVXwithMemcpy, (void *)th1Args);
      pthread_create(&th2, nullptr, applyFilterAVXwithMemcpy, (void *)th2Args);

      pthread_join(th0, nullptr);
      pthread_join(th1, nullptr);
      pthread_join(th2, nullptr);

      cycStop = rdtscll();
      double diff = cycStop - cycStart;
      double diffPerPixel = diff / (output->width * output->height);
      sum += diffPerPixel;
      samples++;
      fprintf(stderr, "Took %f cycles to process, or %f cycles per pixel\n",
              diff, diff / (input->width * input->height));

      // double sample = applyFilter(filter, input, output);
      // sum += sample;
      // samples++;
      cs1300bmp_writefile((char *)outputFilename.c_str(), output);
    }

    delete input;
    delete output;
  }
  fprintf(stdout, "Average cycles per sample is %f\n", sum / samples);


  sum = 0.0;
  samples = 0;
  cout << endl << "**********************************************************************" << endl;
  cout << "Using AVX and Duff's Device for copying each row" << endl;
  for (int inNum = 2; inNum < argc; inNum++)
  {
    string inputFilename = argv[inNum];
    string outputFilename = "filtered-" + filterOutputName + "-" + inputFilename;
    struct cs1300bmp *input = new struct cs1300bmp;
    struct cs1300bmp *output = new struct cs1300bmp;

    posix_memalign(reinterpret_cast<void **>(&input), 32, sizeof(cs1300bmp));
    posix_memalign(reinterpret_cast<void **>(&output), 32, sizeof(cs1300bmp));

    int ok = cs1300bmp_readfile((char *)inputFilename.c_str(), input);

    if (ok)
    {
      pthread_t th0;
      struct args *th0Args = (struct args *)malloc(sizeof(args));
      th0Args->filter = filter;
      th0Args->input = input;
      th0Args->output = output;
      th0Args->plane = 0;

      pthread_t th1;
      struct args *th1Args = (struct args *)malloc(sizeof(args));
      th1Args->filter = filter;
      th1Args->input = input;
      th1Args->output = output;
      th1Args->plane = 1;

      pthread_t th2;
      struct args *th2Args = (struct args *)malloc(sizeof(args));
      th2Args->filter = filter;
      th2Args->input = input;
      th2Args->output = output;
      th2Args->plane = 2;

      long long cycStart, cycStop;
      cycStart = rdtscll();

      pthread_create(&th0, nullptr, applyFilterAVXwithDuffDevice, (void *)th0Args);
      pthread_create(&th1, nullptr, applyFilterAVXwithDuffDevice, (void *)th1Args);
      pthread_create(&th2, nullptr, applyFilterAVXwithDuffDevice, (void *)th2Args);

      pthread_join(th0, nullptr);
      pthread_join(th1, nullptr);
      pthread_join(th2, nullptr);

      cycStop = rdtscll();
      double diff = cycStop - cycStart;
      double diffPerPixel = diff / (output->width * output->height);
      sum += diffPerPixel;
      samples++;
      fprintf(stderr, "Took %f cycles to process, or %f cycles per pixel\n",
              diff, diff / (input->width * input->height));

      // double sample = applyFilter(filter, input, output);
      // sum += sample;
      // samples++;
      cs1300bmp_writefile((char *)outputFilename.c_str(), output);
    }

    delete input;
    delete output;
  }
  fprintf(stdout, "Average cycles per sample is %f\n", sum / samples);


  sum = 0.0;
  samples = 0;
  cout << endl << "**********************************************************************" << endl;
  cout << "Using SSE and assign pixels one by one" << endl;
  for (int inNum = 2; inNum < argc; inNum++)
  {
    string inputFilename = argv[inNum];
    string outputFilename = "filtered-" + filterOutputName + "-" + inputFilename;
    struct cs1300bmp *input = new struct cs1300bmp;
    struct cs1300bmp *output = new struct cs1300bmp;

    posix_memalign(reinterpret_cast<void **>(&input), 32, sizeof(cs1300bmp));
    posix_memalign(reinterpret_cast<void **>(&output), 32, sizeof(cs1300bmp));

    int ok = cs1300bmp_readfile((char *)inputFilename.c_str(), input);

    if (ok)
    {
      pthread_t th0;
      struct args *th0Args = (struct args *)malloc(sizeof(args));
      th0Args->filter = filter;
      th0Args->input = input;
      th0Args->output = output;
      th0Args->plane = 0;

      pthread_t th1;
      struct args *th1Args = (struct args *)malloc(sizeof(args));
      th1Args->filter = filter;
      th1Args->input = input;
      th1Args->output = output;
      th1Args->plane = 1;

      pthread_t th2;
      struct args *th2Args = (struct args *)malloc(sizeof(args));
      th2Args->filter = filter;
      th2Args->input = input;
      th2Args->output = output;
      th2Args->plane = 2;

      long long cycStart, cycStop;
      cycStart = rdtscll();

      pthread_create(&th0, nullptr, applyFilterPlaneSSE, (void *)th0Args);
      pthread_create(&th1, nullptr, applyFilterPlaneSSE, (void *)th1Args);
      pthread_create(&th2, nullptr, applyFilterPlaneSSE, (void *)th2Args);

      pthread_join(th0, nullptr);
      pthread_join(th1, nullptr);
      pthread_join(th2, nullptr);

      cycStop = rdtscll();
      double diff = cycStop - cycStart;
      double diffPerPixel = diff / (output->width * output->height);
      sum += diffPerPixel;
      samples++;
      fprintf(stderr, "Took %f cycles to process, or %f cycles per pixel\n",
              diff, diff / (input->width * input->height));

      // double sample = applyFilter(filter, input, output);
      // sum += sample;
      // samples++;
      cs1300bmp_writefile((char *)outputFilename.c_str(), output);
    }

    delete input;
    delete output;
  }
  fprintf(stdout, "Average cycles per sample is %f\n", sum / samples);


  sum = 0.0;
  samples = 0;
  cout << endl << "**********************************************************************" << endl;
  cout << "With out using SSE or AVX, and assign pixels one by one" << endl;
  for (int inNum = 2; inNum < argc; inNum++)
  {
    string inputFilename = argv[inNum];
    string outputFilename = "filtered-" + filterOutputName + "-" + inputFilename;
    struct cs1300bmp *input = new struct cs1300bmp;
    struct cs1300bmp *output = new struct cs1300bmp;

    posix_memalign(reinterpret_cast<void **>(&input), 32, sizeof(cs1300bmp));
    posix_memalign(reinterpret_cast<void **>(&output), 32, sizeof(cs1300bmp));

    int ok = cs1300bmp_readfile((char *)inputFilename.c_str(), input);

    if (ok)
    {
      pthread_t th0;
      struct args *th0Args = (struct args *)malloc(sizeof(args));
      th0Args->filter = filter;
      th0Args->input = input;
      th0Args->output = output;
      th0Args->plane = 0;

      pthread_t th1;
      struct args *th1Args = (struct args *)malloc(sizeof(args));
      th1Args->filter = filter;
      th1Args->input = input;
      th1Args->output = output;
      th1Args->plane = 1;

      pthread_t th2;
      struct args *th2Args = (struct args *)malloc(sizeof(args));
      th2Args->filter = filter;
      th2Args->input = input;
      th2Args->output = output;
      th2Args->plane = 2;

      long long cycStart, cycStop;
      cycStart = rdtscll();

      pthread_create(&th0, nullptr, applyFilterNormal, (void *)th0Args);
      pthread_create(&th1, nullptr, applyFilterNormal, (void *)th1Args);
      pthread_create(&th2, nullptr, applyFilterNormal, (void *)th2Args);

      pthread_join(th0, nullptr);
      pthread_join(th1, nullptr);
      pthread_join(th2, nullptr);

      cycStop = rdtscll();
      double diff = cycStop - cycStart;
      double diffPerPixel = diff / (output->width * output->height);
      sum += diffPerPixel;
      samples++;
      fprintf(stderr, "Took %f cycles to process, or %f cycles per pixel\n",
              diff, diff / (input->width * input->height));

      // double sample = applyFilter(filter, input, output);
      // sum += sample;
      // samples++;
      cs1300bmp_writefile((char *)outputFilename.c_str(), output);
    }

    delete input;
    delete output;
  }
  fprintf(stdout, "Average cycles per sample is %f\n", sum / samples);
}

struct Filter *
readFilter(string filename)
{
  ifstream input(filename.c_str());

  if (!input.bad())
  {
    int size = 0;
    input >> size;
    Filter *filter = new Filter(size);
    int div;
    input >> div;
    filter->setDivisor(div);
    for (int i = 0; i < size; i++)
    {
      for (int j = 0; j < size; j++)
      {
        int value;
        input >> value;
        filter->set(i, j, value);
      }
    }
    return filter;
  }
  else
  {
    cerr << "Bad input in readFilter:" << filename << endl;
    exit(-1);
  }
}

/*
double
applyFilter(struct Filter *filter, cs1300bmp *input, cs1300bmp *output)
{

  long long cycStart, cycStop;

  cycStart = rdtscll();

  output->width = input->width;
  output->height = input->height;

  int inWidBou = input->width - 1;
  int inHeiBou = input->height - 1;
  short filterDivi = filter->getDivisor();
  short filterArray[FILTER_SIZE][FILTER_SIZE];

  #pragma omp parallel for
  for (short i = 0; i < FILTER_SIZE; i++)
  {
    filterArray[i][0] = filter->get(i, 0);
    filterArray[i][1] = filter->get(i, 1);
    filterArray[i][2] = filter->get(i, 2);
  }

  #pragma omp parallel for
  for (int plane = 0; plane < 3; plane++)
  {
    for (int row = 1; row < inHeiBou; row++)
    {
      short lastRow = row - 1;
      short nextRow = row + 1;

      for (int col = 1; col < inWidBou; col++)
      {
        short lastCol = col - 1;
        short nextCol = col + 1;

        short rowA = (input->color[plane][lastRow][lastCol] * filterArray[0][0]) + (input->color[plane][lastRow][col] * filterArray[0][1]) + (input->color[plane][lastRow][nextCol] * filterArray[0][2]);
        short rowB = (input->color[plane][row][lastCol] * filterArray[1][0]) + (input->color[plane][row][col] * filterArray[1][1]) + (input->color[plane][row][nextCol] * filterArray[1][2]);
        short rowC = (input->color[plane][nextRow][lastCol] * filterArray[2][0]) + (input->color[plane][nextRow][col] * filterArray[2][1]) + (input->color[plane][nextRow][nextCol] * filterArray[2][2]);

        short result = filterDivi > 1 ? ((rowA + rowB + rowC) / filterDivi) : (rowA + rowB + rowC);
        output->color[plane][row][col] = ((result >= 0 && result <= 255) ? result : (result < 0 ? 0 : 255));
      }
    }
  }

  cycStop = rdtscll();
  double diff = cycStop - cycStart;
  double diffPerPixel = diff / (output->width * output->height);
  fprintf(stderr, "Took %f cycles to process, or %f cycles per pixel\n",
          diff, diff / (output->width * output->height));
  return diffPerPixel;
}
*.

/*
double
applyFilter(struct Filter *filter, cs1300bmp *input, cs1300bmp *output)
{

  long long cycStart, cycStop;

  cycStart = rdtscll();

  output->width = input->width;
  output->height = input->height;

  int inWidBou = input->width - 1;
  int inHeiBou = input->height - 1;
  short filterDivi = filter->getDivisor();
  short filterArray[FILTER_SIZE][FILTER_SIZE];
  struct cs1300bmp target;

  #pragma omp parallel for
  for (short i = 0; i < FILTER_SIZE; i++)
  {
    filterArray[i][0] = filter->get(i, 0);
    filterArray[i][1] = filter->get(i, 1);
    filterArray[i][2] = filter->get(i, 2);
  }

  #pragma omp parallel for
  for (int row = 1; row < inHeiBou; row++)
  {
    short lastRow = row - 1;
    short nextRow = row + 1;

    for (int col = 1; col < inWidBou; col++)
    {
      short lastCol = col - 1;
      short nextCol = col + 1;

      short rowA0 = (input->color[0][lastRow][lastCol] * filterArray[0][0]) + (input->color[0][lastRow][col] * filterArray[0][1]) + (input->color[0][lastRow][nextCol] * filterArray[0][2]);
      short rowB0 = (input->color[0][row][lastCol] * filterArray[1][0]) + (input->color[0][row][col] * filterArray[1][1]) + (input->color[0][row][nextCol] * filterArray[1][2]);
      short rowC0 = (input->color[0][nextRow][lastCol] * filterArray[2][0]) + (input->color[0][nextRow][col] * filterArray[2][1]) + (input->color[0][nextRow][nextCol] * filterArray[2][2]);

      short rowA1 = (input->color[1][lastRow][lastCol] * filterArray[0][0]) + (input->color[1][lastRow][col] * filterArray[0][1]) + (input->color[1][lastRow][nextCol] * filterArray[0][2]);
      short rowB1 = (input->color[1][row][lastCol] * filterArray[1][0]) + (input->color[1][row][col] * filterArray[1][1]) + (input->color[1][row][nextCol] * filterArray[1][2]);
      short rowC1 = (input->color[1][nextRow][lastCol] * filterArray[2][0]) + (input->color[1][nextRow][col] * filterArray[2][1]) + (input->color[1][nextRow][nextCol] * filterArray[2][2]);

      short rowA2 = (input->color[2][lastRow][lastCol] * filterArray[0][0]) + (input->color[2][lastRow][col] * filterArray[0][1]) + (input->color[2][lastRow][nextCol] * filterArray[0][2]);
      short rowB2 = (input->color[2][row][lastCol] * filterArray[1][0]) + (input->color[2][row][col] * filterArray[1][1]) + (input->color[2][row][nextCol] * filterArray[1][2]);
      short rowC2 = (input->color[2][nextRow][lastCol] * filterArray[2][0]) + (input->color[2][nextRow][col] * filterArray[2][1]) + (input->color[2][nextRow][nextCol] * filterArray[2][2]);


      short result0 = filterDivi > 1 ? ((rowA0 + rowB0 + rowC0) / filterDivi) : (rowA0 + rowB0 + rowC0);
      short result1 = filterDivi > 1 ? ((rowA1 + rowB1 + rowC1) / filterDivi) : (rowA1 + rowB1 + rowC1);
      short result2 = filterDivi > 1 ? ((rowA2 + rowB2 + rowC2) / filterDivi) : (rowA2 + rowB2 + rowC2);

      result0 = ((result0 >= 0 && result0 <= 255) ? result0 : (result0 < 0 ? 0 : 255));
      result1 = ((result1 >= 0 && result1 <= 255) ? result1 : (result1 < 0 ? 0 : 255));
      result2 = ((result2 >= 0 && result2 <= 255) ? result2 : (result2 < 0 ? 0 : 255));

      output->color[0][row][col] = result0;
      output->color[1][row][col] = result1;
      output->color[2][row][col] = result2;
    }
  }

  cycStop = rdtscll();
  double diff = cycStop - cycStart;
  double diffPerPixel = diff / (output->width * output->height);
  fprintf(stderr, "Took %f cycles to process, or %f cycles per pixel\n",
          diff, diff / (output->width * output->height));
  return diffPerPixel;
}
*/

/*
double
applyFilter(struct Filter *filter, cs1300bmp *input, cs1300bmp *output)
{

  long long cycStart, cycStop;

  cycStart = rdtscll();
  
  output->width = input->width;
  output->height = input->height;

  int inWidBou = input->width - 1;
  int inHeiBou = input->height - 1;

  short filterDivi = filter->getDivisor();
  int filterArray[FILTER_SIZE][FILTER_SIZE];

  #pragma omp parallel for
  for (int i = 0; i < FILTER_SIZE; i++)
  {
    filterArray[i][0] = filter->get(i, 0);
    filterArray[i][1] = filter->get(i, 1);
    filterArray[i][2] = filter->get(i, 2);
  }

  const __m128i filterVectorRowA = _mm_setr_epi32(
    filterArray[0][0],
    filterArray[0][1],
    filterArray[0][2],
    0
  );

  const __m128i filterVectorRowB = _mm_setr_epi32(
    filterArray[1][0],
    filterArray[1][1],
    filterArray[1][2],
    0
  );

  const __m128i filterVectorRowC = _mm_setr_epi32(
    filterArray[2][0],
    filterArray[2][1],
    filterArray[2][2],
    0
  );

  //store the result for each row (images in this lab are fairly small, no need of worring about insufficient stack space)
  int resultRowA[4] = {0};
  int resultRowB[4] = {0};
  int resultRowC[4] = {0};

  __m128i *resultVectorRowA = reinterpret_cast<__m128i *>(resultRowA);
  __m128i *resultVectorRowB = reinterpret_cast<__m128i *>(resultRowB);
  __m128i *resultVectorRowC = reinterpret_cast<__m128i *>(resultRowC);

  __m128i *imageVectorRowA = reinterpret_cast<__m128i *>(input->color[0][0]);
  __m128i *imageVectorRowB = reinterpret_cast<__m128i *>(input->color[0][1]);
  __m128i *imageVectorRowC = reinterpret_cast<__m128i *>(input->color[0][2]);

  #pragma omp for simd
  for (int plane = 0; plane < 3; plane++)
  {
    for (int row = 1; row < inHeiBou; row++)
    {
      posix_memalign(reinterpret_cast<void **>(&imageVectorRowA), 4, 128);
      posix_memalign(reinterpret_cast<void **>(&imageVectorRowB), 4, 128);
      posix_memalign(reinterpret_cast<void **>(&imageVectorRowC), 4, 128);

      for (int col = 1; col < inWidBou; col++)
      {
        imageVectorRowA = reinterpret_cast<__m128i *>(&input->color[plane][row - 1][col]);
        imageVectorRowB = reinterpret_cast<__m128i *>(&input->color[plane][row][col]);
        imageVectorRowC = reinterpret_cast<__m128i *>(&input->color[plane][row + 1][col]);

        *resultVectorRowA = _mm_mullo_epi32(*imageVectorRowA, filterVectorRowA);
        *resultVectorRowB = _mm_mullo_epi32(*imageVectorRowB, filterVectorRowB);
        *resultVectorRowC = _mm_mullo_epi32(*imageVectorRowC, filterVectorRowC);

        int rowASum = ((int *)resultVectorRowA)[0] + ((int *)resultVectorRowA)[1] + ((int *)resultVectorRowA)[2];
        int rowBSum = ((int *)resultVectorRowB)[0] + ((int *)resultVectorRowB)[1] + ((int *)resultVectorRowB)[2];
        int rowCSum = ((int *)resultVectorRowC)[0] + ((int *)resultVectorRowC)[1] + ((int *)resultVectorRowC)[2];

        int result = (rowASum + rowBSum + rowCSum) / filterDivi;
        output->color[plane][row][col] = ((result >= 0 && result <= 255) ? result : (result < 0 ? 0 : 255));
      }
    }
  }

  cycStop = rdtscll();

  double diff = cycStop - cycStart;
  double diffPerPixel = diff / (output->width * output->height);
  fprintf(stderr, "Took %f cycles to process, or %f cycles per pixel\n",
          diff, diff / (output->width * output->height));
  return diffPerPixel;
}
*/

/*
double
applyFilter(struct Filter *filter, cs1300bmp *input, cs1300bmp *output)
{

  long long cycStart, cycStop;

  cycStart = rdtscll();
  
  output->width = input->width;
  output->height = input->height;

  int inWidBou = input->width - 1;
  int inHeiBou = input->height - 1;

  short filterDivi = filter->getDivisor();
  int filArr[FILTER_SIZE][FILTER_SIZE];

  #pragma omp parallel for
  for (int i = 0; i < FILTER_SIZE; i++)
  {
    filArr[i][0] = filter->get(i, 0);
    filArr[i][1] = filter->get(i, 1);
    filArr[i][2] = filter->get(i, 2);
  }

  __m256i filVecRowA = _mm256_setr_epi32(
    filArr[0][0], filArr[0][1], filArr[0][2],
    filArr[0][0], filArr[0][1], filArr[0][2],
    filArr[0][0], filArr[0][1]);

  __m256i filVecRowB = _mm256_setr_epi32(
    filArr[1][0], filArr[1][1], filArr[1][2],
    filArr[1][0], filArr[1][1], filArr[1][2],
    filArr[1][0], filArr[1][1]);

  __m256i filVecRowC = _mm256_setr_epi32(
    filArr[2][0], filArr[2][1], filArr[2][2],
    filArr[2][0], filArr[2][1], filArr[2][2],
    filArr[2][0], filArr[2][1]);

  int inWidLoop = (inWidBou + 1) / 6;
  int inWidLeft = inWidLeft % 8;
  
  int resultRows[9][8] = {0};
  int finalRow[input->width];
  __m256i *resVecRowAbias0 = reinterpret_cast<__m256i *>(resultRows[0]);
  __m256i *resVecRowBbias0 = reinterpret_cast<__m256i *>(resultRows[1]);
  __m256i *resVecRowCbias0 = reinterpret_cast<__m256i *>(resultRows[2]);

  __m256i *resVecRowAbias1 = reinterpret_cast<__m256i *>(resultRows[3]);
  __m256i *resVecRowBbias1 = reinterpret_cast<__m256i *>(resultRows[4]);
  __m256i *resVecRowCbias1 = reinterpret_cast<__m256i *>(resultRows[5]);

  __m256i *resVecRowAbias2 = reinterpret_cast<__m256i *>(resultRows[6]);
  __m256i *resVecRowBbias2 = reinterpret_cast<__m256i *>(resultRows[7]);
  __m256i *resVecRowCbias2 = reinterpret_cast<__m256i *>(resultRows[8]);

  #pragma omp for simd
  for (int plane = 0; plane < 3; plane++)
  {
    for (int row = 1; row < inHeiBou; row++)
    {
      int tempAdd[6] = {0};
      int tempResult[6] = {0};

      __m256i *imgVecRowAbias0 = reinterpret_cast<__m256i *>(&input->color[plane][row - 1][0]);
      __m256i *imgVecRowBbias0 = reinterpret_cast<__m256i *>(&input->color[plane][row][0]);
      __m256i *imgVecRowCbias0 = reinterpret_cast<__m256i *>(&input->color[plane][row + 1][0]);

      __m256i *imgVecRowAbias1 = reinterpret_cast<__m256i *>(&input->color[plane][row - 1][1]);
      __m256i *imgVecRowBbias1 = reinterpret_cast<__m256i *>(&input->color[plane][row][1]);
      __m256i *imgVecRowCbias1 = reinterpret_cast<__m256i *>(&input->color[plane][row + 1][1]);

      __m256i *imgVecRowAbias2 = reinterpret_cast<__m256i *>(&input->color[plane][row - 1][2]);
      __m256i *imgVecRowBbias2 = reinterpret_cast<__m256i *>(&input->color[plane][row][2]);
      __m256i *imgVecRowCbias2 = reinterpret_cast<__m256i *>(&input->color[plane][row + 1][2]);

      for (int col = 0; col < inWidLoop; )
      {
        // calculate rows with bias 0
        *resVecRowAbias0 = _mm256_mullo_epi32(*imgVecRowAbias0, filVecRowA);
        *resVecRowBbias0 = _mm256_mullo_epi32(*imgVecRowBbias0, filVecRowB);
        *resVecRowCbias0 = _mm256_mullo_epi32(*imgVecRowCbias0, filVecRowC);

        tempAdd[0] = (resultRows[0][0] + resultRows[0][1] + resultRows[0][2]) + (resultRows[1][0] + resultRows[1][1] + resultRows[1][2]) + (resultRows[2][0] + resultRows[2][1] + resultRows[2][2]);
        tempAdd[1] = (resultRows[0][3] + resultRows[0][4] + resultRows[0][5]) + (resultRows[1][3] + resultRows[1][4] + resultRows[1][5]) + (resultRows[2][3] + resultRows[2][4] + resultRows[2][5]);

        tempResult[0] = tempAdd[0] / filterDivi;
        finalRow[col*6+1] = ((tempResult[0] >= 0 && tempResult[0] <= 255) ? tempResult[0] : (tempResult[0] < 0 ? 0 : 255));

        tempResult[1] = tempAdd[1] / filterDivi;
        finalRow[col*6+4] = ((tempResult[1] >= 0 && tempResult[1] <= 255) ? tempResult[1] : (tempResult[1] < 0 ? 0 : 255));

        // calculate rows with bias 1
        *resVecRowAbias1 = _mm256_mullo_epi32(*imgVecRowAbias1, filVecRowA);
        *resVecRowBbias1 = _mm256_mullo_epi32(*imgVecRowBbias1, filVecRowB);
        *resVecRowCbias1 = _mm256_mullo_epi32(*imgVecRowCbias1, filVecRowC);

        tempAdd[2] = (resultRows[3][0] + resultRows[3][1] + resultRows[3][2]) + (resultRows[4][0] + resultRows[4][1] + resultRows[4][2]) + (resultRows[5][0] + resultRows[5][1] + resultRows[5][2]);
        tempAdd[3] = (resultRows[3][3] + resultRows[3][4] + resultRows[3][5]) + (resultRows[4][3] + resultRows[4][4] + resultRows[4][5]) + (resultRows[5][3] + resultRows[5][4] + resultRows[5][5]);

        tempResult[2] = tempAdd[2] / filterDivi;
        finalRow[col*6+2] = ((tempResult[2] >= 0 && tempResult[2] <= 255) ? tempResult[2] : (tempResult[2] < 0 ? 0 : 255));

        tempResult[3] = tempAdd[3] / filterDivi;
        finalRow[col*6+5] = ((tempResult[3] >= 0 && tempResult[3] <= 255) ? tempResult[3] : (tempResult[3] < 0 ? 0 : 255));

        // calculate rows with bias 2
        *resVecRowAbias2 = _mm256_mullo_epi32(*imgVecRowAbias2, filVecRowA);
        *resVecRowBbias2 = _mm256_mullo_epi32(*imgVecRowBbias2, filVecRowB);
        *resVecRowCbias2 = _mm256_mullo_epi32(*imgVecRowCbias2, filVecRowC);

        tempAdd[4] = (resultRows[6][0] + resultRows[6][1] + resultRows[6][2]) + (resultRows[7][0] + resultRows[7][1] + resultRows[7][2]) + (resultRows[8][0] + resultRows[8][1] + resultRows[8][2]);
        tempAdd[5] = (resultRows[6][3] + resultRows[6][4] + resultRows[6][5]) + (resultRows[7][3] + resultRows[7][4] + resultRows[7][5]) + (resultRows[8][3] + resultRows[8][4] + resultRows[8][5]);

        tempResult[4] = tempAdd[4] / filterDivi;
        finalRow[col*6+3] = ((tempResult[4] >= 0 && tempResult[4] <= 255) ? tempResult[4] : (tempResult[4] < 0 ? 0 : 255));

        tempResult[5] = tempAdd[5] / filterDivi;
        finalRow[col*6+6] = ((tempResult[5] >= 0 && tempResult[5] <= 255) ? tempResult[5] : (tempResult[5] < 0 ? 0 : 255));

        col++;
        imgVecRowAbias0 = reinterpret_cast<__m256i *>(&input->color[plane][row - 1][col*6]);
        imgVecRowBbias0 = reinterpret_cast<__m256i *>(&input->color[plane][row][col*6]);
        imgVecRowCbias0 = reinterpret_cast<__m256i *>(&input->color[plane][row + 1][col*6]);

        imgVecRowAbias1 = reinterpret_cast<__m256i *>(&input->color[plane][row - 1][col*6+1]);
        imgVecRowBbias1 = reinterpret_cast<__m256i *>(&input->color[plane][row][col*6+1]);
        imgVecRowCbias1 = reinterpret_cast<__m256i *>(&input->color[plane][row + 1][col*6+1]);

        imgVecRowAbias2 = reinterpret_cast<__m256i *>(&input->color[plane][row - 1][col*6+2]);
        imgVecRowBbias2 = reinterpret_cast<__m256i *>(&input->color[plane][row][col*6+2]);
        imgVecRowCbias2 = reinterpret_cast<__m256i *>(&input->color[plane][row + 1][col*6+2]);
      }
      //memcpy((void *)output->color[plane][row], (void *)finalRow, input->width * 4);
      char *srcMemReg = (char *)finalRow;
      char *dstMemReg = (char *)output->color[plane][row];
      int jmpIndex = (1024 * 4 + 7) / 8;

      switch (1024 * 4 % 8)
      {
      case 0: do { *dstMemReg++ = *srcMemReg++;
      case 7:      *dstMemReg++ = *srcMemReg++;
      case 6:      *dstMemReg++ = *srcMemReg++;
      case 5:      *dstMemReg++ = *srcMemReg++;
      case 4:      *dstMemReg++ = *srcMemReg++;
      case 3:      *dstMemReg++ = *srcMemReg++;
      case 2:      *dstMemReg++ = *srcMemReg++;
      case 1:      *dstMemReg++ = *srcMemReg++;
            } while (--jmpIndex > 0);
      }
    }
  }

  cycStop = rdtscll();

  double diff = cycStop - cycStart;
  double diffPerPixel = diff / (output->width * output->height);
  fprintf(stderr, "Took %f cycles to process, or %f cycles per pixel\n",
          diff, diff / (output->width * output->height));
  return diffPerPixel;
}
*/


void *applyFilterPlaneSSE(void *arg)
{
  struct Filter *filter = ((args *)arg)->filter;
  cs1300bmp *input = ((args *)arg)->input;
  cs1300bmp *output = ((args *)arg)->output;
  int plane = ((args *)arg)->plane;
  
  output->width = input->width;
  output->height = input->height;

  int inWidBou = input->width - 1;
  int inHeiBou = input->height - 1;

  short filterDivi = filter->getDivisor();
  int filterArray[FILTER_SIZE][FILTER_SIZE];

  for (int i = 0; i < FILTER_SIZE; i++)
  {
    filterArray[i][0] = filter->get(i, 0);
    filterArray[i][1] = filter->get(i, 1);
    filterArray[i][2] = filter->get(i, 2);
  }

  const __m128i filterVectorRowA = _mm_setr_epi32(
    filterArray[0][0],
    filterArray[0][1],
    filterArray[0][2],
    0
  );

  const __m128i filterVectorRowB = _mm_setr_epi32(
    filterArray[1][0],
    filterArray[1][1],
    filterArray[1][2],
    0
  );

  const __m128i filterVectorRowC = _mm_setr_epi32(
    filterArray[2][0],
    filterArray[2][1],
    filterArray[2][2],
    0
  );

  //store the result for each row (images in this lab are fairly small, no need of worring about insufficient stack space)
  int resultRowA[4] = {0};
  int resultRowB[4] = {0};
  int resultRowC[4] = {0};

  __m128i *resultVectorRowA = reinterpret_cast<__m128i *>(resultRowA);
  __m128i *resultVectorRowB = reinterpret_cast<__m128i *>(resultRowB);
  __m128i *resultVectorRowC = reinterpret_cast<__m128i *>(resultRowC);

  __m128i *imageVectorRowA = reinterpret_cast<__m128i *>(input->color[0][0]);
  __m128i *imageVectorRowB = reinterpret_cast<__m128i *>(input->color[0][1]);
  __m128i *imageVectorRowC = reinterpret_cast<__m128i *>(input->color[0][2]);

  
  for (int row = 1; row < inHeiBou; row++)
  {
    posix_memalign(reinterpret_cast<void **>(&imageVectorRowA), 4, 128);
    posix_memalign(reinterpret_cast<void **>(&imageVectorRowB), 4, 128);
    posix_memalign(reinterpret_cast<void **>(&imageVectorRowC), 4, 128);

    for (int col = 1; col < inWidBou; col++)
    {
      imageVectorRowA = reinterpret_cast<__m128i *>(&input->color[plane][row - 1][col]);
      imageVectorRowB = reinterpret_cast<__m128i *>(&input->color[plane][row][col]);
      imageVectorRowC = reinterpret_cast<__m128i *>(&input->color[plane][row + 1][col]);

      *resultVectorRowA = _mm_mullo_epi32(*imageVectorRowA, filterVectorRowA);
      *resultVectorRowB = _mm_mullo_epi32(*imageVectorRowB, filterVectorRowB);
      *resultVectorRowC = _mm_mullo_epi32(*imageVectorRowC, filterVectorRowC);

      int rowASum = ((int *)resultVectorRowA)[0] + ((int *)resultVectorRowA)[1] + ((int *)resultVectorRowA)[2];
      int rowBSum = ((int *)resultVectorRowB)[0] + ((int *)resultVectorRowB)[1] + ((int *)resultVectorRowB)[2];
      int rowCSum = ((int *)resultVectorRowC)[0] + ((int *)resultVectorRowC)[1] + ((int *)resultVectorRowC)[2];

      int result = (rowASum + rowBSum + rowCSum) / filterDivi;
      output->color[plane][row][col] = ((result >= 0 && result <= 255) ? result : (result < 0 ? 0 : 255));
    }
  }
}


void *applyFilterNormal(void *arg)
{
  struct Filter *filter = ((args *)arg)->filter;
  cs1300bmp *input = ((args *)arg)->input;
  cs1300bmp *output = ((args *)arg)->output;
  int plane = ((args *)arg)->plane;

  output->width = input->width;
  output->height = input->height;

  int inWidBou = input->width - 1;
  int inHeiBou = input->height - 1;
  short filterDivi = filter->getDivisor();
  short filterArray[FILTER_SIZE][FILTER_SIZE];

  #pragma omp parallel for
  for (short i = 0; i < FILTER_SIZE; i++)
  {
    filterArray[i][0] = filter->get(i, 0);
    filterArray[i][1] = filter->get(i, 1);
    filterArray[i][2] = filter->get(i, 2);
  }

  #pragma omp parallel for
  for (int row = 1; row < inHeiBou; row++)
  {
    short lastRow = row - 1;
    short nextRow = row + 1;

    for (int col = 1; col < inWidBou; col++)
    {
      short lastCol = col - 1;
      short nextCol = col + 1;

      short rowA = (input->color[plane][lastRow][lastCol] * filterArray[0][0]) + (input->color[plane][lastRow][col] * filterArray[0][1]) + (input->color[plane][lastRow][nextCol] * filterArray[0][2]);
      short rowB = (input->color[plane][row][lastCol] * filterArray[1][0]) + (input->color[plane][row][col] * filterArray[1][1]) + (input->color[plane][row][nextCol] * filterArray[1][2]);
      short rowC = (input->color[plane][nextRow][lastCol] * filterArray[2][0]) + (input->color[plane][nextRow][col] * filterArray[2][1]) + (input->color[plane][nextRow][nextCol] * filterArray[2][2]);

      short result = filterDivi > 1 ? ((rowA + rowB + rowC) / filterDivi) : (rowA + rowB + rowC);
      output->color[plane][row][col] = ((result >= 0 && result <= 255) ? result : (result < 0 ? 0 : 255));
    }
  }
}


void *applyFilterAVXwithMemcpy(void *arg)
{
  struct Filter *filter = ((args *)arg)->filter;
  cs1300bmp *input = ((args *)arg)->input;
  cs1300bmp *output = ((args *)arg)->output;
  int plane = ((args *)arg)->plane;
  
  output->width = input->width;
  output->height = input->height;

  int inWidBou = input->width - 1;
  int inHeiBou = input->height - 1;

  short filterDivi = filter->getDivisor();
  int filArr[FILTER_SIZE][FILTER_SIZE];

  #pragma omp parallel for
  for (int i = 0; i < FILTER_SIZE; i++)
  {
    filArr[i][0] = filter->get(i, 0);
    filArr[i][1] = filter->get(i, 1);
    filArr[i][2] = filter->get(i, 2);
  }

  __m256i filVecRowA = _mm256_setr_epi32(
    filArr[0][0], filArr[0][1], filArr[0][2],
    filArr[0][0], filArr[0][1], filArr[0][2],
    filArr[0][0], filArr[0][1]);

  __m256i filVecRowB = _mm256_setr_epi32(
    filArr[1][0], filArr[1][1], filArr[1][2],
    filArr[1][0], filArr[1][1], filArr[1][2],
    filArr[1][0], filArr[1][1]);

  __m256i filVecRowC = _mm256_setr_epi32(
    filArr[2][0], filArr[2][1], filArr[2][2],
    filArr[2][0], filArr[2][1], filArr[2][2],
    filArr[2][0], filArr[2][1]);

  int inWidLoop = (inWidBou + 1) / 6;
  int inWidLeft = inWidLeft % 8;
  
  int resultRows[9][8] = {0};
  int finalRow[input->width];
  __m256i *resVecRowAbias0 = reinterpret_cast<__m256i *>(resultRows[0]);
  __m256i *resVecRowBbias0 = reinterpret_cast<__m256i *>(resultRows[1]);
  __m256i *resVecRowCbias0 = reinterpret_cast<__m256i *>(resultRows[2]);

  __m256i *resVecRowAbias1 = reinterpret_cast<__m256i *>(resultRows[3]);
  __m256i *resVecRowBbias1 = reinterpret_cast<__m256i *>(resultRows[4]);
  __m256i *resVecRowCbias1 = reinterpret_cast<__m256i *>(resultRows[5]);

  __m256i *resVecRowAbias2 = reinterpret_cast<__m256i *>(resultRows[6]);
  __m256i *resVecRowBbias2 = reinterpret_cast<__m256i *>(resultRows[7]);
  __m256i *resVecRowCbias2 = reinterpret_cast<__m256i *>(resultRows[8]);

  #pragma omp for simd
  for (int row = 1; row < inHeiBou; row++)
  {
    int tempAdd[6] = {0};
    int tempResult[6] = {0};

    __m256i *imgVecRowAbias0 = reinterpret_cast<__m256i *>(&input->color[plane][row - 1][0]);
    __m256i *imgVecRowBbias0 = reinterpret_cast<__m256i *>(&input->color[plane][row][0]);
    __m256i *imgVecRowCbias0 = reinterpret_cast<__m256i *>(&input->color[plane][row + 1][0]);

    __m256i *imgVecRowAbias1 = reinterpret_cast<__m256i *>(&input->color[plane][row - 1][1]);
    __m256i *imgVecRowBbias1 = reinterpret_cast<__m256i *>(&input->color[plane][row][1]);
    __m256i *imgVecRowCbias1 = reinterpret_cast<__m256i *>(&input->color[plane][row + 1][1]);

    __m256i *imgVecRowAbias2 = reinterpret_cast<__m256i *>(&input->color[plane][row - 1][2]);
    __m256i *imgVecRowBbias2 = reinterpret_cast<__m256i *>(&input->color[plane][row][2]);
    __m256i *imgVecRowCbias2 = reinterpret_cast<__m256i *>(&input->color[plane][row + 1][2]);

    for (int col = 0; col < inWidLoop; )
    {
      // calculate rows with bias 0
      *resVecRowAbias0 = _mm256_mullo_epi32(*imgVecRowAbias0, filVecRowA);
      *resVecRowBbias0 = _mm256_mullo_epi32(*imgVecRowBbias0, filVecRowB);
      *resVecRowCbias0 = _mm256_mullo_epi32(*imgVecRowCbias0, filVecRowC);

      tempAdd[0] = (resultRows[0][0] + resultRows[0][1] + resultRows[0][2]) + (resultRows[1][0] + resultRows[1][1] + resultRows[1][2]) + (resultRows[2][0] + resultRows[2][1] + resultRows[2][2]);
      tempAdd[1] = (resultRows[0][3] + resultRows[0][4] + resultRows[0][5]) + (resultRows[1][3] + resultRows[1][4] + resultRows[1][5]) + (resultRows[2][3] + resultRows[2][4] + resultRows[2][5]);

      tempResult[0] = tempAdd[0] / filterDivi;
      finalRow[col*6+1] = ((tempResult[0] >= 0 && tempResult[0] <= 255) ? tempResult[0] : (tempResult[0] < 0 ? 0 : 255));

      tempResult[1] = tempAdd[1] / filterDivi;
      finalRow[col*6+4] = ((tempResult[1] >= 0 && tempResult[1] <= 255) ? tempResult[1] : (tempResult[1] < 0 ? 0 : 255));

      // calculate rows with bias 1
      *resVecRowAbias1 = _mm256_mullo_epi32(*imgVecRowAbias1, filVecRowA);
      *resVecRowBbias1 = _mm256_mullo_epi32(*imgVecRowBbias1, filVecRowB);
      *resVecRowCbias1 = _mm256_mullo_epi32(*imgVecRowCbias1, filVecRowC);

      tempAdd[2] = (resultRows[3][0] + resultRows[3][1] + resultRows[3][2]) + (resultRows[4][0] + resultRows[4][1] + resultRows[4][2]) + (resultRows[5][0] + resultRows[5][1] + resultRows[5][2]);
      tempAdd[3] = (resultRows[3][3] + resultRows[3][4] + resultRows[3][5]) + (resultRows[4][3] + resultRows[4][4] + resultRows[4][5]) + (resultRows[5][3] + resultRows[5][4] + resultRows[5][5]);

      tempResult[2] = tempAdd[2] / filterDivi;
      finalRow[col*6+2] = ((tempResult[2] >= 0 && tempResult[2] <= 255) ? tempResult[2] : (tempResult[2] < 0 ? 0 : 255));

      tempResult[3] = tempAdd[3] / filterDivi;
      finalRow[col*6+5] = ((tempResult[3] >= 0 && tempResult[3] <= 255) ? tempResult[3] : (tempResult[3] < 0 ? 0 : 255));

      // calculate rows with bias 2
      *resVecRowAbias2 = _mm256_mullo_epi32(*imgVecRowAbias2, filVecRowA);
      *resVecRowBbias2 = _mm256_mullo_epi32(*imgVecRowBbias2, filVecRowB);
      *resVecRowCbias2 = _mm256_mullo_epi32(*imgVecRowCbias2, filVecRowC);

      tempAdd[4] = (resultRows[6][0] + resultRows[6][1] + resultRows[6][2]) + (resultRows[7][0] + resultRows[7][1] + resultRows[7][2]) + (resultRows[8][0] + resultRows[8][1] + resultRows[8][2]);
      tempAdd[5] = (resultRows[6][3] + resultRows[6][4] + resultRows[6][5]) + (resultRows[7][3] + resultRows[7][4] + resultRows[7][5]) + (resultRows[8][3] + resultRows[8][4] + resultRows[8][5]);

      tempResult[4] = tempAdd[4] / filterDivi;
      finalRow[col*6+3] = ((tempResult[4] >= 0 && tempResult[4] <= 255) ? tempResult[4] : (tempResult[4] < 0 ? 0 : 255));

      tempResult[5] = tempAdd[5] / filterDivi;
      finalRow[col*6+6] = ((tempResult[5] >= 0 && tempResult[5] <= 255) ? tempResult[5] : (tempResult[5] < 0 ? 0 : 255));

      col++;
      imgVecRowAbias0 = reinterpret_cast<__m256i *>(&input->color[plane][row - 1][col*6]);
      imgVecRowBbias0 = reinterpret_cast<__m256i *>(&input->color[plane][row][col*6]);
      imgVecRowCbias0 = reinterpret_cast<__m256i *>(&input->color[plane][row + 1][col*6]);

      imgVecRowAbias1 = reinterpret_cast<__m256i *>(&input->color[plane][row - 1][col*6+1]);
      imgVecRowBbias1 = reinterpret_cast<__m256i *>(&input->color[plane][row][col*6+1]);
      imgVecRowCbias1 = reinterpret_cast<__m256i *>(&input->color[plane][row + 1][col*6+1]);

      imgVecRowAbias2 = reinterpret_cast<__m256i *>(&input->color[plane][row - 1][col*6+2]);
      imgVecRowBbias2 = reinterpret_cast<__m256i *>(&input->color[plane][row][col*6+2]);
      imgVecRowCbias2 = reinterpret_cast<__m256i *>(&input->color[plane][row + 1][col*6+2]);
    }
    memcpy((void *)output->color[plane][row], (void *)finalRow, input->width * 4);
  }

}


void *applyFilterAVXwithDuffDevice(void *arg)
{
  struct Filter *filter = ((args *)arg)->filter;
  cs1300bmp *input = ((args *)arg)->input;
  cs1300bmp *output = ((args *)arg)->output;
  int plane = ((args *)arg)->plane;
  
  output->width = input->width;
  output->height = input->height;

  int inWidBou = input->width - 1;
  int inHeiBou = input->height - 1;

  short filterDivi = filter->getDivisor();
  int filArr[FILTER_SIZE][FILTER_SIZE];

  #pragma omp parallel for
  for (int i = 0; i < FILTER_SIZE; i++)
  {
    filArr[i][0] = filter->get(i, 0);
    filArr[i][1] = filter->get(i, 1);
    filArr[i][2] = filter->get(i, 2);
  }

  __m256i filVecRowA = _mm256_setr_epi32(
    filArr[0][0], filArr[0][1], filArr[0][2],
    filArr[0][0], filArr[0][1], filArr[0][2],
    filArr[0][0], filArr[0][1]);

  __m256i filVecRowB = _mm256_setr_epi32(
    filArr[1][0], filArr[1][1], filArr[1][2],
    filArr[1][0], filArr[1][1], filArr[1][2],
    filArr[1][0], filArr[1][1]);

  __m256i filVecRowC = _mm256_setr_epi32(
    filArr[2][0], filArr[2][1], filArr[2][2],
    filArr[2][0], filArr[2][1], filArr[2][2],
    filArr[2][0], filArr[2][1]);

  int inWidLoop = (inWidBou + 1) / 6;
  int inWidLeft = inWidLeft % 8;
  
  int resultRows[9][8] = {0};
  int finalRow[input->width];
  __m256i *resVecRowAbias0 = reinterpret_cast<__m256i *>(resultRows[0]);
  __m256i *resVecRowBbias0 = reinterpret_cast<__m256i *>(resultRows[1]);
  __m256i *resVecRowCbias0 = reinterpret_cast<__m256i *>(resultRows[2]);

  __m256i *resVecRowAbias1 = reinterpret_cast<__m256i *>(resultRows[3]);
  __m256i *resVecRowBbias1 = reinterpret_cast<__m256i *>(resultRows[4]);
  __m256i *resVecRowCbias1 = reinterpret_cast<__m256i *>(resultRows[5]);

  __m256i *resVecRowAbias2 = reinterpret_cast<__m256i *>(resultRows[6]);
  __m256i *resVecRowBbias2 = reinterpret_cast<__m256i *>(resultRows[7]);
  __m256i *resVecRowCbias2 = reinterpret_cast<__m256i *>(resultRows[8]);

  #pragma omp for simd
  for (int row = 1; row < inHeiBou; row++)
  {
    int tempAdd[6] = {0};
    int tempResult[6] = {0};

    __m256i *imgVecRowAbias0 = reinterpret_cast<__m256i *>(&input->color[plane][row - 1][0]);
    __m256i *imgVecRowBbias0 = reinterpret_cast<__m256i *>(&input->color[plane][row][0]);
    __m256i *imgVecRowCbias0 = reinterpret_cast<__m256i *>(&input->color[plane][row + 1][0]);

    __m256i *imgVecRowAbias1 = reinterpret_cast<__m256i *>(&input->color[plane][row - 1][1]);
    __m256i *imgVecRowBbias1 = reinterpret_cast<__m256i *>(&input->color[plane][row][1]);
    __m256i *imgVecRowCbias1 = reinterpret_cast<__m256i *>(&input->color[plane][row + 1][1]);

    __m256i *imgVecRowAbias2 = reinterpret_cast<__m256i *>(&input->color[plane][row - 1][2]);
    __m256i *imgVecRowBbias2 = reinterpret_cast<__m256i *>(&input->color[plane][row][2]);
    __m256i *imgVecRowCbias2 = reinterpret_cast<__m256i *>(&input->color[plane][row + 1][2]);

    for (int col = 0; col < inWidLoop; )
    {
      // calculate rows with bias 0
      *resVecRowAbias0 = _mm256_mullo_epi32(*imgVecRowAbias0, filVecRowA);
      *resVecRowBbias0 = _mm256_mullo_epi32(*imgVecRowBbias0, filVecRowB);
      *resVecRowCbias0 = _mm256_mullo_epi32(*imgVecRowCbias0, filVecRowC);

      tempAdd[0] = (resultRows[0][0] + resultRows[0][1] + resultRows[0][2]) + (resultRows[1][0] + resultRows[1][1] + resultRows[1][2]) + (resultRows[2][0] + resultRows[2][1] + resultRows[2][2]);
      tempAdd[1] = (resultRows[0][3] + resultRows[0][4] + resultRows[0][5]) + (resultRows[1][3] + resultRows[1][4] + resultRows[1][5]) + (resultRows[2][3] + resultRows[2][4] + resultRows[2][5]);

      tempResult[0] = tempAdd[0] / filterDivi;
      finalRow[col*6+1] = ((tempResult[0] >= 0 && tempResult[0] <= 255) ? tempResult[0] : (tempResult[0] < 0 ? 0 : 255));

      tempResult[1] = tempAdd[1] / filterDivi;
      finalRow[col*6+4] = ((tempResult[1] >= 0 && tempResult[1] <= 255) ? tempResult[1] : (tempResult[1] < 0 ? 0 : 255));

      // calculate rows with bias 1
      *resVecRowAbias1 = _mm256_mullo_epi32(*imgVecRowAbias1, filVecRowA);
      *resVecRowBbias1 = _mm256_mullo_epi32(*imgVecRowBbias1, filVecRowB);
      *resVecRowCbias1 = _mm256_mullo_epi32(*imgVecRowCbias1, filVecRowC);

      tempAdd[2] = (resultRows[3][0] + resultRows[3][1] + resultRows[3][2]) + (resultRows[4][0] + resultRows[4][1] + resultRows[4][2]) + (resultRows[5][0] + resultRows[5][1] + resultRows[5][2]);
      tempAdd[3] = (resultRows[3][3] + resultRows[3][4] + resultRows[3][5]) + (resultRows[4][3] + resultRows[4][4] + resultRows[4][5]) + (resultRows[5][3] + resultRows[5][4] + resultRows[5][5]);

      tempResult[2] = tempAdd[2] / filterDivi;
      finalRow[col*6+2] = ((tempResult[2] >= 0 && tempResult[2] <= 255) ? tempResult[2] : (tempResult[2] < 0 ? 0 : 255));

      tempResult[3] = tempAdd[3] / filterDivi;
      finalRow[col*6+5] = ((tempResult[3] >= 0 && tempResult[3] <= 255) ? tempResult[3] : (tempResult[3] < 0 ? 0 : 255));

      // calculate rows with bias 2
      *resVecRowAbias2 = _mm256_mullo_epi32(*imgVecRowAbias2, filVecRowA);
      *resVecRowBbias2 = _mm256_mullo_epi32(*imgVecRowBbias2, filVecRowB);
      *resVecRowCbias2 = _mm256_mullo_epi32(*imgVecRowCbias2, filVecRowC);

      tempAdd[4] = (resultRows[6][0] + resultRows[6][1] + resultRows[6][2]) + (resultRows[7][0] + resultRows[7][1] + resultRows[7][2]) + (resultRows[8][0] + resultRows[8][1] + resultRows[8][2]);
      tempAdd[5] = (resultRows[6][3] + resultRows[6][4] + resultRows[6][5]) + (resultRows[7][3] + resultRows[7][4] + resultRows[7][5]) + (resultRows[8][3] + resultRows[8][4] + resultRows[8][5]);

      tempResult[4] = tempAdd[4] / filterDivi;
      finalRow[col*6+3] = ((tempResult[4] >= 0 && tempResult[4] <= 255) ? tempResult[4] : (tempResult[4] < 0 ? 0 : 255));

      tempResult[5] = tempAdd[5] / filterDivi;
      finalRow[col*6+6] = ((tempResult[5] >= 0 && tempResult[5] <= 255) ? tempResult[5] : (tempResult[5] < 0 ? 0 : 255));

      col++;
      imgVecRowAbias0 = reinterpret_cast<__m256i *>(&input->color[plane][row - 1][col*6]);
      imgVecRowBbias0 = reinterpret_cast<__m256i *>(&input->color[plane][row][col*6]);
      imgVecRowCbias0 = reinterpret_cast<__m256i *>(&input->color[plane][row + 1][col*6]);

      imgVecRowAbias1 = reinterpret_cast<__m256i *>(&input->color[plane][row - 1][col*6+1]);
      imgVecRowBbias1 = reinterpret_cast<__m256i *>(&input->color[plane][row][col*6+1]);
      imgVecRowCbias1 = reinterpret_cast<__m256i *>(&input->color[plane][row + 1][col*6+1]);

      imgVecRowAbias2 = reinterpret_cast<__m256i *>(&input->color[plane][row - 1][col*6+2]);
      imgVecRowBbias2 = reinterpret_cast<__m256i *>(&input->color[plane][row][col*6+2]);
      imgVecRowCbias2 = reinterpret_cast<__m256i *>(&input->color[plane][row + 1][col*6+2]);
    }
    // using duff's device is slower than memcpy under this condition
    char *srcMemReg = (char *)finalRow;
    char *dstMemReg = (char *)output->color[plane][row];
    int jmpIndex = (1024 * 4 + 7) / 8;

    switch (1024 * 4 % 8)
    {
    case 0: do { *dstMemReg++ = *srcMemReg++;
    case 7:      *dstMemReg++ = *srcMemReg++;
    case 6:      *dstMemReg++ = *srcMemReg++;
    case 5:      *dstMemReg++ = *srcMemReg++;
    case 4:      *dstMemReg++ = *srcMemReg++;
    case 3:      *dstMemReg++ = *srcMemReg++;
    case 2:      *dstMemReg++ = *srcMemReg++;
    case 1:      *dstMemReg++ = *srcMemReg++;
          } while (--jmpIndex > 0);
    }
  }

}