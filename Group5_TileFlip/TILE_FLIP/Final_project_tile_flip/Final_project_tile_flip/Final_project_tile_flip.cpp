// Final_project_tile_flip.cpp : Defines the entry point for the console application.
//

// TileFlip.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
#include <windows.h>
#pragma comment(lib, "Winmm.lib")


#define LEVEL_FILE			"Input/level_file.txt"
#define MAX_COLUMN				8
#define MAX_ROW					8
#define COLOR_NAME_SIZE			40
#define NO_OF_COLOR				12
#define INFINITY				999999


/* Voice Detection defines */
#define VECTOR_SIZE				12
#define FRAME_SIZE				320
#define NORMALIZE_VALUE			5000
#define CODEBOOK_SIZE			32
#define MAX_FILE_NAME_SIZE		200
#define NO_OF_OBSERVATION		125
#define NO_OF_SAMPLE			320 + (NO_OF_OBSERVATION - 1) * 80
#define FORWARD_SAMPLE			(((NO_OF_SAMPLE) / 2) - 1)
#define BACKWARD_SAMPLE			((NO_OF_SAMPLE) / 2)
#define INPUT_CEPSTRAL			"dump_Cdash_test.txt"
#define INT_MAX					999999
#define NO_OF_DIGITS			11
#define SILENCE_FILE				"Input/silence_file.txt"
#define INPUT_FILE					"Input/Input_File/184101035_3_27.txt"
#define CODEBOOK_FILE_NAME			"Input/codebook.txt"
#define HAMMING_WINDOW				"Input/Hamming_window.txt"
#define HMM_OBSERVATION_SEQUENCE	"Input/HMM_OBSERVATION_SEQUENCE.txt"
/* File to be used in the Implementation */

#define HMM_AIJ_FINAL				"HMM_AIJ_FINAL.txt"
#define HMM_BJK_FINAL				"HMM_BJK_FINAL.txt"
#define HMM_PII_FINAL				"HMM_PII_FINAL.txt"

#define HMM_ALPHA_MATRIX			"HMM_ALPHA_MATRIX.txt"
#define HMM_BETA_MATRIX				"HMM_BETA_MATRIX.txt"
#define HMM_STATE_SEQUENCE			"HMM_STATE_SEQUENCE_VITERBI.txt"
//#define INTERVAL_SIZE				7040
//#define TRIM_WINDOW_SHIFT			250
#define NO_OF_DIGIT					9
#define DIGIT_UTTERANCE				30



/* Debug print Enable/Disable */
#define DEBUG_PRINT


#define DEBUG_OBSERVATION_SEQUENCE_PRINT
#define DEBUG_STATE_SEQUENCE_PRINT

/* No of states in the model (N) */
#define NO_OF_STATE					5	

/* No of distict observation symbols per state (M) */
#define NO_OF_OBSERVATION_SYMBOL	32				

/* No of observation for a given observation sequence (T) */

/* Threshold value to update Bjk matrix */
#define BJK_THRESHOLD				pow(10.0, -30.0)

/* Deciding which gamma to use, Solution 2 or Solution 3, though both are same. Solution 3 gamma do not a last row */
#define OLD_GAMMA_SOLUTION2

/* Number of iteration to use for training the model */
#define NO_OF_ITERATION_TO_TRAIN	20

int TRIM_WINDOW_SHIFT = 250;
int INTERVAL_SIZE = 320 + (NO_OF_OBSERVATION - 1) * 80;


/* Global */
FILE *codebook_fp, *input_fp_cepstral;
FILE *input_fp, *operation_fp, *dump_Ri, *dump_Ai, *dump_Ci, *silence_fp, *sample_file_fp, *hamming_file, *dump_Cdash;
FILE *dump_Cdash_reference, *deleteit;
long double codebook[CODEBOOK_SIZE][VECTOR_SIZE] = { 0 };
long double array_input_cepstral[NO_OF_OBSERVATION][VECTOR_SIZE] = { 0 };
long int array_observation_sequence[NO_OF_OBSERVATION + 1] = { 0 };


long double array_alpha_ti[NO_OF_STATE + 1][NO_OF_OBSERVATION + 1] = { 0 };
long double array_beta_ti[NO_OF_STATE + 1][NO_OF_OBSERVATION + 1] = { 0 };
long double array_aij[NO_OF_STATE + 1][NO_OF_STATE + 1] = { 0 };
long double array_bjk[NO_OF_STATE + 1][NO_OF_OBSERVATION_SYMBOL + 1] = { 0 };
long double array_pii[NO_OF_STATE + 1] = { 0 };
long double array_delta_ti[NO_OF_STATE + 1][NO_OF_OBSERVATION + 1] = { 0 };
long double array_XI_ijt[NO_OF_STATE + 1][NO_OF_STATE + 1][NO_OF_OBSERVATION + 1] = { 0 };
long double array_gamma_ti[NO_OF_STATE + 1][NO_OF_OBSERVATION + 1] = { 0 };
long double array_summation[NO_OF_OBSERVATION + 1] = { 0 };
long int array_psi_ti[NO_OF_STATE + 1][NO_OF_OBSERVATION + 1] = { 0 };
long int array_state_sequence[NO_OF_OBSERVATION + 1] = { 0 };
long double b_summation[NO_OF_STATE + 1] = { 0 };
long double Pstar = 0.0;
long int QstarT = 0;
long double probability_of_O_given_model = 0.0;


FILE *output_fp_alpha, *output_fp_beta, *output_fp_gamma, *output_state_sequence, *output_state_sequence_gamma;
FILE *output_fp_aij, *output_fp_bjk, *output_fp_pii;
FILE *input_fp_aij, *input_fp_bjk, *input_fp_pii, *input_fp_observation_sequence, *input_fp_alpha, *input_fp_beta;


char HMM_AIJ[MAX_FILE_NAME_SIZE];
char HMM_BJK[MAX_FILE_NAME_SIZE];
char HMM_PII[MAX_FILE_NAME_SIZE];
char input_file_name[MAX_FILE_NAME_SIZE];

/* Function Declaration */
double find_DC_shift(FILE *silence_input_fp);
long long int count_no_of_sample(FILE *input_fp);
long long int apply_DC_shift_Normalize(FILE *input_fp, FILE *operation_fp, double shift_value, double ratio);
void do_autocorelation(double arr[], long double arr_Ri[], long long int length, FILE *dump_Ri);
void do_durbin(long double arr_Ri[], FILE *dump_Ai, long double arr_Ai[]);
void do_capstral(long double arr_Ai[], long double arr_Ri[], long double arr_Ci[], FILE *dump_Ci);
void apply_sine_window(long double arr_Cdash[], long double arr_Ci[], FILE *dump_Cdash);
void copy_to_array(FILE *operation_fp, double arr[], long long int *length);
void find_frames(double sample_array[], double arr[], long long int *start, long long int *end, long long int length, FILE *sample_file_fp);
void get_hamming_window(FILE *hamming_file, double hamming_window[]);
void apply_hamming_window(double hamming_window[], double frame_array[], double sample_array[], long long int sliding_index);
long double tohkure_distance(FILE *dump_Cdash, FILE *dump_Cdash_reference);
void print_info();
void read_Model();
void forward_step1_initialization();
void forward_step2_induction();
void forward_step3_termination();
void print_Observation_Sequence();
void debug_print();
void read_Alpha_Beta();
int check_status[12] = { 0 };


/* Function Definiton */
/*****************************************************
*  Name			: find_DC_shift						 *
*													 *
*  Arguments	: silence file pointer				 *
*													 *
*  Description	: Find DC shift value for file		 *
*													 *
*  Return		: DC shift value					 *
*													 *
*****************************************************/
double find_DC_shift(FILE *silence_input_fp)
{
	rewind(silence_input_fp);
	double DC_SHIFT_VALUE = 0.0;
	long long int no_of_sample = 0;
	long long int count = 0;
	long double value = 0.0;
	long double sum = 0;
	if (silence_input_fp == NULL)
	{
		printf("Could not open Silence file for DC SHIFT calculation\n");
		return -9999;
	}
	while (1)
	{
		if (feof(silence_input_fp))
		{
			break;
		}
		count++;
		fscanf(silence_input_fp, "%Lf", &value);

		/* Discarding first 200 sample */
		if (count > 200)
		{
			sum = sum + value;
			no_of_sample++;
		}

	}
	//printf("\nSum is %Lf, number of sample %ld\n", sum, no_of_sample);
	DC_SHIFT_VALUE = sum / no_of_sample;

	return DC_SHIFT_VALUE;
}

/*****************************************************
*  Name			: count_no_of_sample				 *
*													 *
*  Arguments	: input file pointer				 *
*													 *
*  Description	: Find number of sample in file		 *
*													 *
*  Return		: No of samples in file				 *
*													 *
*****************************************************/

long long int count_no_of_sample(FILE *input_fp)
{
	rewind(input_fp);
	long long int no_of_sample = 0;
	double value = 0;

	if (input_fp == NULL)
	{
		printf("Could not open Input file for sample calculation\n");
		return -1;
	}

	while (1)
	{
		if (feof(input_fp))
		{
			break;
		}
		fscanf(input_fp, "%lf", &value);
		no_of_sample++;
	}

	return no_of_sample;
}

/*****************************************************
*  Name			: find_max							 *
*													 *
*  Arguments	: input file pointer				 *
*													 *
*  Description	: Find max sample value in file		 *
*													 *
*  Return		: Find max value in file			 *
*													 *
*****************************************************/
double find_max(FILE *input_fp)
{
	rewind(input_fp);
	double max = 0;
	double value = 0;

	if (input_fp == NULL)
	{
		printf("Could not open Input file for max calculation\n");
		return -1;
	}

	while (1)
	{
		if (feof(input_fp))
		{
			break;
		}
		fscanf(input_fp, "%lf", &value);
		if (max < abs(value))
		{
			max = abs(value);
		}
	}
	return max;
}

/*****************************************************
*  Name			: apply_DC_shift_Normalize			 *
*													 *
*  Arguments	: input file pointer, shift value, 	 *
*				  normalization ratio, output file   *
*				  pointer							 *
*													 *
*  Description	: Apply DC shift, normalize the file *
*				  create new file with normalized    *
*				  value								 *
*													 *
*  Return		: 0 on success and -1 on failure	 *
*													 *
*****************************************************/
long long int apply_DC_shift_Normalize(FILE *input_fp, FILE *operation_fp, double shift_value, double ratio)
{

	rewind(input_fp);
	rewind(operation_fp);
	double value = 0.0;
	if (input_fp == NULL)
	{
		printf("Could not open Input file for DC shift\n");
		return -1;
	}
	if (operation_fp == NULL)
	{
		printf("Could not open output file for DC shift\n");
		return -1;
	}

	while (1)
	{
		if (feof(input_fp))
		{
			break;
		}
		fscanf(input_fp, "%lf", &value);
		value = value - shift_value;
		value = (double)value * ratio;
		fprintf(operation_fp, "%lf\n", value);
	}
	return 0;
}

/*****************************************************
*  Name			: do_autocorelation					 *
*													 *
*  Arguments	: Sample array file, array for Ri to *
*				  length of input file,Ri file       *
*				  pointer							 *
*													 *
*  Description	: Find Ri and dump into file	     *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void do_autocorelation(double arr[], long double arr_Ri[], long long int length, FILE *dump_Ri)
{
	long long int p = VECTOR_SIZE;
	long long int i = 0, j = 0;
	long long int frame_count = 0;

	for (i = 0; i < 13; i++)
	{
		arr_Ri[i] = 0;
	}

	frame_count = length / FRAME_SIZE;

	for (i = 0; i <= p; i++)
	{
		for (j = 0; j < FRAME_SIZE - i; j++)
		{
			arr_Ri[i] = arr[j] * arr[j + i] + arr_Ri[i];
		}
	}

	for (i = 0; i <= p; i++)
	{
		//printf("R %lld is : %Lf\n", i, arr_Ri[i]);
		fprintf(dump_Ri, "%Lf\t", arr_Ri[i]);
	}
	fprintf(dump_Ri, "\n");
}

/*****************************************************
*  Name			: do_durbin							 *
*													 *
*  Arguments	: Ri array file, array for Ri, Ai 	 *
*				  array								 *
*													 *
*  Description	: Find Ai and dump into file	     *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void do_durbin(long double arr_Ri[], FILE *dump_Ai, long double arr_Ai[])
{
	long long int i = 0;
	for (i = 0; i < 13; i++)
	{
		arr_Ai[i] = 0;
	}

	long double Energy[20] = { 0 };
	long double alpha[20][20] = { 0 };
	long long int j = 0;
	long double coefficient_K[20] = { 0 };
	long long int p = VECTOR_SIZE;

	if (arr_Ri[0] < 60)
	{
		//printf("Error: Energy cannot be so less\n");
		return;
	}

	/* Assigning R[0] to E[0] */
	Energy[0] = arr_Ri[0];

	for (i = 1; i <= p; i++)
	{
		if (i == 1)
		{
			coefficient_K[i] = arr_Ri[1] / arr_Ri[0];
			alpha[i][i] = coefficient_K[i];
		}
		else
		{
			for (j = 1; j <= i - 1; j++)
			{
				coefficient_K[i] = alpha[j][i - 1] * arr_Ri[i - j] + coefficient_K[i];
			}
			coefficient_K[i] = (arr_Ri[i] - coefficient_K[i]) / Energy[i - 1];
			alpha[i][i] = coefficient_K[i];

			if (i > 1)
			{
				for (j = 1; j <= i - 1; j++)
				{
					alpha[j][i] = alpha[j][i - 1] - coefficient_K[i] * alpha[i - j][i - 1];
				}
			}

		}
		Energy[i] = (1 - coefficient_K[i] * coefficient_K[i]) * Energy[i - 1];
	}
	for (j = 0; j <= p; j++)
	{
		arr_Ai[j] = alpha[j][12];
	}

	for (j = 1; j <= 12; j++)
	{
		//printf("A %lld is: %Lf\n",j, arr_Ai[j]);
		fprintf(dump_Ai, "%Lf\t", arr_Ai[j]);
	}
	fprintf(dump_Ai, "\n");
}

/*****************************************************
*  Name			: do_capstral						 *
*													 *
*  Arguments	: Ai array file, array for Ai, Ci 	 *
*				  array								 *
*													 *
*  Description	: Find Ci and dump into file	     *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void do_capstral(long double arr_Ai[], long double arr_Ri[], long double arr_Ci[], FILE *dump_Ci)
{
	long long int i = 0;
	long long int p = VECTOR_SIZE;
	long long int m = 0, k = 0;
	long double x = 0.0;

	for (i = 0; i < 14; i++)
	{
		arr_Ci[i] = 0;
	}

	arr_Ci[0] = 2.0 * log(arr_Ri[0]) / log(2.0);
	//printf("gain factor :%lf\n\n", arr_Ci[0]);
	for (m = 1; m <= p; m++)
	{
		x = 0;
		for (k = 1; k <= m - 1; k++)
		{
			x = ((double)k / m) * arr_Ci[k] * arr_Ai[m - k] + x;
		}
		arr_Ci[m] = arr_Ai[m] + x;
	}

	for (i = 1; i <= 12; i++)
	{
		//printf("C %lld is: %Lf\n", i, arr_Ci[i]);
		fprintf(dump_Ci, "%Lf\t", arr_Ci[i]);
	}
	fprintf(dump_Ci, "\n");
}

/*****************************************************
*  Name			: apply_sine_window					 *
*													 *
*  Arguments	: Ci array file, array for Cprime 	 *
*				  cprime file pointer				 *
*													 *
*  Description	: Find Ci and dump into file	     *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void apply_sine_window(long double arr_Cdash[], long double arr_Ci[], FILE *dump_Cdash)
{
	long long int m = 0;
	long long int p = VECTOR_SIZE;
	long double window[14] = { 0 };
	for (m = 1; m <= p; m++)
	{
		window[m] = 1.0 + 6.0 * sin(22.0 / 7.0 * m / 12.0);
		arr_Cdash[m] = window[m] * arr_Ci[m];
		fprintf(dump_Cdash, "%Lf\t", arr_Cdash[m]);
	}
	fprintf(dump_Cdash, "\n", arr_Cdash[m]);
}

/*****************************************************
*  Name			: copy_to_array						 *
*													 *
*  Arguments	: file Pointer, array, length		 *
*													 *
*  Description	: Copy file data to array.		     *
*													 *
*  Return		: Nothing, but update length of file *
*													 *
*****************************************************/
void copy_to_array(FILE *operation_fp, double arr[], long long int *length)
{
	double value = 0.0;
	long long int index = 0;

	if (operation_fp == NULL)
	{
		printf("Could not open output for copying in array\n");
		return;
	}
	rewind(operation_fp);

	while (1)
	{
		if (feof(operation_fp))
		{
			break;
		}
		fscanf(operation_fp, "%lf", &value);
		arr[index++] = value;
	}
	*length = index;
}

/*****************************************************
*  Name			: find_frames						 *
*													 *
*  Arguments	: file Pointer, array to copy, start,*
*				  end pointer.						 *
*													 *
*  Description	: Find the stable frame for vowel    *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
#if 0
void find_frames(double sample_array[], double arr[], long long int *start, long long int *end, long long int length, FILE *sample_file_fp)
{

	long long int i = 0, j;
	long long int index = 0;
	double max_value = 0;
	long long int x, y;
	long double value = 0.0;
	long long int count = 0;


	for (i = 0; i < length; i++)
	{
		if (max_value < abs(arr[i]))
		{
			max_value = abs(arr[i]);
			j = i;
		}
	}

	x = j - (long long int)BACKWARD_SAMPLE + 1;
	y = j + (long long int)FORWARD_SAMPLE + 1;
	//printf("max value is %lf and index is %ld\n", max_value, j);
	//printf("Forward is :%lld\nBackward is :%lld\n",(long long int)FORWARD_SAMPLE, (long long int)BACKWARD_SAMPLE);
	//printf("Start = %lld\nEnd = %lld\nDiffernce in start and end: %lld\n",x, y, (long long int)y - x);
	//printf("Length %d\n",length);

	if (length < 7040)
	{
		printf("The number of sample in files are very less");
	}

	if (x < 0)
	{
		x = 1;
		y = INTERVAL_SIZE;
	}
	if (y >= length)
	{
		y = length - 2;
		x = y - INTERVAL_SIZE + 1;
	}

	for (i = x; i < y; i++)
	{
		sample_array[index++] = arr[i];
		fprintf(sample_file_fp, "%lf\n", sample_array[index - 1]);
	}
	sample_array[index++] = arr[i];
	fprintf(sample_file_fp, "%lf", sample_array[index - 1]);
	//printf("First value is %Lf\n", arr[x]);
	//printf("last value is %Lf\n", arr[y]);
	*start = x;
	*end = y;

}

#endif
void find_frames(double sample_array[], double arr[], long long int *start, long long int *end, long long int length, FILE *sample_file_fp)
{

	long long int i = 0, j;
	long long int index = 0;
	double max_value = 0;
	long long int x = 0, y = 0;;
	long double value = 0.0;
	int count = 0;
	long double energy = 0;
	long double intermediate_enery = 0;
	int start_marker = 0;
	int end_marker = 0;
	//int TRIM_WINDOW_SHIFT1 = 250;
	//int INTERVAL_SIZE = 9840;

	length = length - 1;
	//	printf("The length of input obtained is %ld\n", length);
	for (i = 0; i < length - 1; i++)
	{
		count++;
		//printf("value of iiiiii is .......................................... %d .......\n", i);
		intermediate_enery += 0.01 * arr[i] * arr[i];
		if (count == INTERVAL_SIZE)
		{
			//printf("count ::%ld\n ",count);
			count = 0;
			if (energy < intermediate_enery)
			{
				start_marker = i - (INTERVAL_SIZE - 1);
				end_marker = i;
				energy = intermediate_enery;
			}
			//printf("value of i before is %d ... ", i);
			i = (i + TRIM_WINDOW_SHIFT) - INTERVAL_SIZE;
			//printf("value of i after is %d ... ", i);
			intermediate_enery = 0;
		}
	}

	//	printf(" Start marker is %d \n", start_marker);
	//printf("End marker is %d\n", end_marker);

	if (x > 1000)
	{
		x = start_marker;
		y = end_marker;
	}
	else
	{
		x = start_marker;
		y = end_marker;

	}





#if 0
	for (i = 0; i < length; i++)
	{
		if (max_value < abs(arr[i]))
		{
			max_value = abs(arr[i]);
			j = i;
		}
	}

	x = j - (long long int)BACKWARD_SAMPLE + 1;
	y = j + (long long int)FORWARD_SAMPLE + 1;
	//printf("max value is %lf and index is %ld\n", max_value, j);
	//printf("Forward is :%lld\nBackward is :%lld\n",(long long int)FORWARD_SAMPLE, (long long int)BACKWARD_SAMPLE);
	//printf("Start = %lld\nEnd = %lld\nDiffernce in start and end: %lld\n",x, y, (long long int)y - x);
	//printf("Length %d\n",length);

	if (length < INTERVAL_SIZE)
	{
		printf("The number of sample in files are very less");
	}

	if (x < 0)
	{
		x = 1;
		y = INTERVAL_SIZE;
	}
	if (y >= length)
	{
		y = length - 2;
		x = y - INTERVAL_SIZE + 1;
	}
#endif

	for (i = x; i < y; i++)
	{
		sample_array[index++] = arr[i];
		fprintf(sample_file_fp, "%lf\n", sample_array[index - 1]);
	}
	sample_array[index++] = arr[i];
	fprintf(sample_file_fp, "%lf", sample_array[index - 1]);
	//printf("First value is %Lf\n", arr[x]);
	//printf("last value is %Lf\n", arr[y]);
	*start = x;
	*end = y;


}





/*****************************************************
*  Name			: get_hamming_window				 *
*													 *
*  Arguments	: Hamming file Pointer, array to copy*
*													 *
*  Description	: Copy the hamming window to array   *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void get_hamming_window(FILE *hamming_file, double hamming_window[])
{
	rewind(hamming_file);
	double value = 0.0;
	long long int index = 0;
	long long int i = 0;

	while (1)
	{
		if (feof(hamming_file))
		{
			break;
		}
		fscanf(hamming_file, "%lf", &value);
		hamming_window[index++] = value;
	}
}

/*****************************************************
*  Name			: apply_hamming_window				 *
*													 *
*  Arguments	: Hamming array, sample array, slide *
*				  index								 *
*													 *
*  Description	: Apply hamming window to input frame*
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void apply_hamming_window(double hamming_window[], double frame_array[], double sample_array[], long long int sliding_index)
{
	long long int i = 0;
	for (i = 0; i < FRAME_SIZE; i++)
	{
		frame_array[i] = hamming_window[i] * sample_array[sliding_index + i];
	}

}

/*****************************************************
*  Name			: print_codeBook					 *
*													 *
*  Arguments	: codebook							 *
*													 *
*  Description	: print codebook					 *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void print_codeBook(long double codebook[CODEBOOK_SIZE][VECTOR_SIZE])
{
	long long int i, j;

	for (i = 0; i < CODEBOOK_SIZE; i++)
	{
		for (j = 0; j < VECTOR_SIZE; j++)
		{
			printf("%Lf\t", codebook[i][j]);
		}
		printf("\n");
	}
}

/*****************************************************
*  Name			: tohkure_distance					 *
*													 *
*  Arguments	: cprime file pointer, referece file *
*				  pointer							 *
*													 *
*  Description	: Find the tohkure distance of vector*
*													 *
*  Return		: Distance caluculated				 *
*													 *
*****************************************************/

double tohkure_distance(long double training_vector[], long double codebook_vector[])
{
	double tohkura_weights[13] = { 1.0, 3.0, 7.0, 13.0, 19.0, 22.0, 25.0, 33.0, 42.0, 50.0, 56.0, 61.0 };
	long long int i = 0;
	double reference_value = 0.0;
	double test_value = 0.0;
	double distance = 0;

	for (i = 0; i < VECTOR_SIZE; i++)
	{
		reference_value = codebook_vector[i];
		test_value = training_vector[i];
		distance = tohkura_weights[i] * (reference_value - test_value) * (reference_value - test_value) + distance;
	}

	return distance;

}

/*****************************************************
*  Name			: find_observation_sequence			 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Find observation sequence for data *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void find_observation_sequence()
{
	long long int i, j;
	long double distance = INT_MAX;
	long double tohkure = 0;
	input_fp_cepstral = fopen(INPUT_CEPSTRAL, "r+");

	for (i = 0; i < NO_OF_OBSERVATION; i++)
	{
		for (j = 0; j < VECTOR_SIZE; j++)
		{
			fscanf(input_fp_cepstral, "%Lf", &array_input_cepstral[i][j]);
		}
	}

	for (i = 0; i < NO_OF_OBSERVATION; i++)
	{
		distance = INT_MAX;
		for (j = 0; j < CODEBOOK_SIZE; j++)
		{
			tohkure = tohkure_distance(array_input_cepstral[i], codebook[j]);
			if (distance > tohkure)
			{
				array_observation_sequence[i + 1] = j + 1;
				distance = tohkure;
			}
		}
	}
}

/*****************************************************
*  Name			: read_codebook						 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Read the codebook in array		 *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void read_codebook()
{
	long long int i, j;

	codebook_fp = fopen(CODEBOOK_FILE_NAME, "r+");
	for (i = 0; i < CODEBOOK_SIZE; i++)
	{
		for (j = 0; j < VECTOR_SIZE; j++)
		{
			fscanf(codebook_fp, "%Lf", &codebook[i][j]);
		}
	}
}

/* Function Definition */
/*****************************************************
*  Name			: print_info						 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Print what code actually does.     *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void print_info()
{

}

/*****************************************************
*  Name			: read_Model						 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Read A, B, PI from file			 *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void read_Model()
{
	long long int i, j;

	/* Open Aij, Bjk, PIi, Observation Sequence */
	input_fp_aij = fopen(HMM_AIJ, "r+");
	input_fp_bjk = fopen(HMM_BJK, "r+");
	input_fp_pii = fopen(HMM_PII, "r+");
	//	input_fp_observation_sequence = fopen(HMM_OBSERVATION_SEQUENCE, "r+");

	/* Read Aij */
	for (i = 1; i <= NO_OF_STATE; i++)
	{
		for (j = 1; j <= NO_OF_STATE; j++)
		{
			fscanf(input_fp_aij, "%Le", &array_aij[i][j]);
		}
	}

	/* Read Bjk */
	for (i = 1; i <= NO_OF_STATE; i++)
	{
		for (j = 1; j <= NO_OF_OBSERVATION_SYMBOL; j++)
		{
			fscanf(input_fp_bjk, "%Le", &array_bjk[i][j]);
		}
	}

	/* Read PIi*/
	for (i = 1; i <= NO_OF_STATE; i++)
	{
		fscanf(input_fp_pii, "%Le", &array_pii[i]);
	}

	/* Read Observation Sequence */
	/*	for(i = 1 ; i <= NO_OF_OBSERVATION; i++)
	{
	fscanf(input_fp_observation_sequence, "%ld", &array_observation_sequence[i]);
	}
	*/
	/* Close file */
	fclose(input_fp_aij);
	fclose(input_fp_bjk);
	fclose(input_fp_pii);
	//	fclose(input_fp_observation_sequence);

}

/*****************************************************
*  Name			: forward_step1_initialization		 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Initialize alpha matrix			 *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void forward_step1_initialization()
{
	long long int i, j, o1;
	for (i = 1; i <= NO_OF_STATE; i++)
	{
		o1 = array_observation_sequence[i];
		array_alpha_ti[i][1] = array_pii[i] * array_bjk[i][o1];
	}
}

/*****************************************************
*  Name			: forward_step2_induction			 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Update alpha matrix				 *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void forward_step2_induction()
{
	long long int i, j, t, ot;
	long double temp_value = 0;

	/* Open alpa and beta file to dump values */
	output_fp_alpha = fopen(HMM_ALPHA_MATRIX, "w+");

	for (t = 1; t <= NO_OF_OBSERVATION - 1; t++)
	{
		for (j = 1; j <= NO_OF_STATE; j++)
		{
			temp_value = 0;
			for (i = 1; i <= NO_OF_STATE; i++)
			{
				temp_value += array_alpha_ti[i][t] * array_aij[i][j];
			}
			ot = array_observation_sequence[t + 1];
			array_alpha_ti[j][t + 1] = temp_value * array_bjk[j][ot];
		}
	}

	for (i = 1; i <= NO_OF_OBSERVATION; i++)
	{
		for (j = 1; j <= NO_OF_STATE; j++)
		{
			fprintf(output_fp_alpha, "%0.30Le\t", array_alpha_ti[j][i]);
		}
		fprintf(output_fp_alpha, "\n");
	}

	fclose(output_fp_alpha);
}

/*****************************************************
*  Name			: forward_step3_termination			 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Find probability_of_O_given_model	 *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void forward_step3_termination()
{
	long int i;

	probability_of_O_given_model = 0.0;
	for (i = 1; i <= NO_OF_STATE; i++)
	{
		probability_of_O_given_model += array_alpha_ti[i][NO_OF_OBSERVATION];
	}

}


/*****************************************************
*  Name			: print_Observation_Sequence		 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Print input Obs sequence.			 *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void print_Observation_Sequence()
{
	long long int i = 1;
	for (i = 1; i <= NO_OF_OBSERVATION; i++)
	{
		//printf("%ld\t", array_observation_sequence[i]);

	}
	//Sleep(2000);
	//printf("\n");
}

/*****************************************************
*  Name			: debug_print						 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Print all intermdiate matrices	 *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/




















void print_current_grid();
void initialize_id_array();
void initialize_image_color_array();
void play_game(long int current_size);
void detect_color(long int current_size);
//void detect_color()

long int max_column = 2;
long int max_row = 2;
long int color_name_size = 40;
long int no_of_color = 12;
long int overall_score = 0;


int id_array[MAX_COLUMN][MAX_ROW];
char image_array[MAX_COLUMN][MAX_ROW][COLOR_NAME_SIZE];//= {"CAT", "DOG", "HORSE", "MOUSE"};// "HYNA", "TIGER"};
char color_array[MAX_COLUMN][MAX_ROW][COLOR_NAME_SIZE];//= {"ORANGE", "YELLOW", "PINK", "WHITE"};//, "PURPLE", "INDIGO", "GOLDEN", "BLACK", "MAGENTA", "RED", "SILVER", "BROWN"};

char sample_color[NO_OF_COLOR][COLOR_NAME_SIZE] = { "ORANGE", "YELLOW", "PINK", "WHITE", "PURPLE", "INDIGO", "GOLDEN", "BLACK", "MAGENTA", "RED", "SILVER", "BROWN" };
//char sample_image[NO_OF_COLOR][COLOR_NAME_SIZE] = { "CAT", "DOG", "HORSE", "MOUSE", "HYNA", "TIGER", "CAT", "DOG", "HORSE", "MOUSE", "HYNA", "TIGER" };//

char sample_image[NO_OF_COLOR][COLOR_NAME_SIZE] = { "@", "!", "#", "$", "%", "^", "*", "&", "HYNA", "MOUSE", "HYNA", "TIGER" };

long int color_recognized_from_voice = 0;
char color_recognized_from_voice_name[100] = { 0 };

long int answer = 0;
char input_color[COLOR_NAME_SIZE];
long int score = 0;
long int id1, id2;
long int x, y;
FILE *level_fp;











#define N 5
#define M 32
#define T 125
long double alpha[T + 1][N + 1] = { 0 }, pi[N + 1] = { 0 }, b[N + 1][M + 1] = { 0 }, a[N + 1][N + 1] = { 0 }, prob_alpha[13] = { 0 };
int O[T + 1] = { 0 };
long double sum_alpha = 0.0;
FILE *alpha_open = fopen("alpha.txt", "w");
FILE *pi_file = fopen("Input/pi.txt", "r");
char name_a[13][100] = { "Input/a_0.txt", "Input/a_1.txt", "Input/a_2.txt", "Input/a_3.txt", "Input/a_4.txt", "Input/a_5.txt", "Input/a_6.txt", "Input/a_7.txt", "Input/a_8.txt", "Input/a_9.txt", "Input/a_10.txt", "Input/a_11.txt" };
char name_b[13][100] = { "Input/b_0.txt", "Input/b_1.txt", "Input/b_2.txt", "Input/b_3.txt", "Input/b_4.txt", "Input/b_5.txt", "Input/b_6.txt", "Input/b_7.txt", "Input/b_8.txt", "Input/b_9.txt", "Input/b_10.txt", "Input/b_11.txt" };

void dcshift(char silence_file[], char input_file[], char dc_file[])
{
	int n = 0;
	double avg, sum = 0.0, x, a, newa;
	FILE *in_file = fopen(silence_file, "r");
	FILE *read_file = fopen(input_file, "r");
	while (!feof(in_file))
	{
		fscanf(in_file, "%lf\n", &x);
		sum = sum + x;
		n++;
	}
	avg = sum / n;
	FILE *out_file = fopen(dc_file, "w");
	while (!feof(read_file))
	{
		fscanf(read_file, "%lf\n", &a);
		newa = a - avg;
		//  printf("%lf\n", newa);
		fprintf(out_file, "%lf\n", newa);
	}
	fclose(in_file);
	fclose(out_file);
}
//calculating normalisation
void normalisation(char dc_file1[], char normal_file[])
{
	float i, max = 0.0, ratio, newi;
	FILE *dc_file = fopen(dc_file1, "r");
	while (!feof(dc_file))
	{
		fscanf(dc_file, "%f\n", &i);
		if (abs(i) >= max)
		{
			max = abs(i);
		}
	}
	ratio = 5000 / max;
	FILE *out_file = fopen(normal_file, "w");
	rewind(dc_file);
	while (!feof(dc_file))
	{
		fscanf(dc_file, "%f\n", &i);
		newi = i*ratio;
		//printf("%f\n", newi);
		fprintf(out_file, "%f\n", newi);
	}
	//printf("inside normalization\n");
	fclose(dc_file);
	fclose(out_file);
}
//triming the function
void triming(char normal_file3[], char trim_file[])
{
	long long int array[100000] = { 0 }, a = 0, i, n = 0, j = 0, k, r = 0, start, end, t, count = 0;
	long double e, energy = 0, max = 0, x;
	FILE *actual_file = fopen(normal_file3, "r");
	FILE *trim_file1 = fopen(trim_file, "w");

	while (!feof(actual_file))
	{
		fscanf(actual_file, "%lf\n", &x);
		array[a++] = x;
		n++;
		//printf("%lld\n", x);
	}
	for (i = 0; i < n; i++)
	{
		//printf("%lld\n", array[i]);
	}

	for (k = 1; k <= n / 500; k++)
	{
		energy = 0;
		count = j;
		for (j; j < (9920 + r); j++)
		{
			energy = energy + array[j] * array[j];
		}

		//printf("%Lf\n", energy);
		if (energy >= max)
		{
			max = energy;
			start = count;
			end = count + 9920;
			//printf("%lld\n %lld\n", start,end);
		}
		//printf("%lld\n %lld\n", start, end);
		j = j - 9452;
		r = r + 500;
	}
	//printf("%lld\n %lld\n", start, end);
	for (t = start; t <= end; t++)
	{
		//printf("%lld\n", array[t]);
		fprintf(trim_file1, "%lld\n", array[t]);
	}
	//printf("trim is working");
	fclose(actual_file);
	fclose(trim_file1);
	//printf("trim is working 1");
}
void recognition()
{
	//printf("\nstart");
	int  j;
	int i;
	int A = 0;
	int t = 0;
	int l = 0;
	int m = 0;
	int h = 0;
	int v = 0;
	int d = 0;
	int f = 0;

	long double a;
	long double p;
	long double u;
	long double value;
	long double cptest;
	long double di;
	long double x = 0;
	long double sum = 0;
	long double toku = 0;
	//long double values are intialized.
	//printf("before array");

	long double s[15000] = { 0 };
	long double samp[30000] = { 0 };
	long double ham[15000] = { 0 };
	long double R[10000] = { 0 };
	long double E[1000] = { 0 };
	long double K[10000] = { 0 };
	long double X[100][100] = { 0 };
	//printf("hi");
	long double W[1000] = { 0 };
	long double C[5000] = { 0 };
	long double w[5000] = { 0 };
	long double cp[5000] = { 0 };
	long double cpt[5000] = { 0 };
	long double dis[300] = { 0 };
	long double codebook[33][13] = { 0 };
	long double all_cprime[151][13] = { 0 };
	long double weight[500] = { 1.0, 3.0, 7.0, 13.0, 19.0, 22.0, 25.0, 33.0, 42.0, 50.0, 56.0, 61.0 };
	//printf("\nhello");
	FILE *codebook_refernce = fopen("Input/codebook.txt", "r");

	//printf("\nhello");

	FILE *open_file = fopen("trim.txt", "r");
	FILE *observation_file = fopen("observation.txt", "a+");
	//FILE *testing_observation = fopen("test_obs.txt", "a+");
	//open_file contains the testing file name which can be expicitly changed as per our requirement depending in the vowel we have to test.
	FILE *distance = fopen("184101008_test_di.txt", "w");
	//A file for distances have been made.
	fclose(distance);
	distance = fopen("184101008_test_di.txt", "a+");
	FILE *minimum = fopen("184101008_test_min.txt", "w");
	fclose(minimum);
	minimum = fopen("184101008_test_min.txt", "a+");
	//It contains the minimum distance among all the distances.
	FILE *cptesting = fopen("184101008_test_cpi.txt", "w+");
	//It writes the cprimes in the file of testing samples.
	FILE *rif = fopen("184101008_test_ri.txt", "w+");
	//It writes the RI's of testing file.
	FILE *frame_file = fopen("184101008_test_ai.txt", "w+");
	//It gives the AI's of testing file.
	FILE *coef_file = fopen("test_ci.txt", "w+");
	//It contains the cepstral coefficient.
	FILE *out_file = fopen("184101008_test_hamming.txt", "w+");
	//printf("After init\n");

	while (!feof(open_file))
	{
		fscanf(open_file, "%lf\n", &p);
		s[l++] = p;
	}
	//Read the testing file and store the values in an array.

	long double hi = 2.0*(22.0 / 7.0);
	for (int n = 0; n <= 319; n++)
	{
		ham[n] = 0.54 - 0.46*(cos((hi *(double)n) / (double)319));
		fprintf(out_file, "%lf\n", ham[n]);
	}
	//applying the hamming window on the sample.
	for (int f = 0; f < 125; f++)
	{
		// Running the full for loop for 5 frames.
		for (int n = h; n <= 319 + h; n++)
		{
			samp[n] = ham[v] * s[n];
			v++;
		}
		//sample vales are multiplied by hamming window resulting in new sample values.
		for (j = 0; j <= 12; j++)
		{
			for (i = h; i <= 319 + h - t; i++)
			{
				a = samp[i] * samp[i + t];
				R[j] = R[j] + a;
			}
			//calculating RI's per frame.
			fprintf(rif, "%lf\n", R[j]);
			t++;
		}
		fprintf(rif, "\n\n");
		t = 0;

		//Calculation of AI's using durbins algorithm.
		E[0] = R[0];
		K[1] = R[1] / R[0];
		X[1][1] = K[1];
		E[1] = (1 - (K[1] * K[1]))*E[0];
		for (int i = 2; i <= 12; i++)
		{
			u = 0.0;
			for (int j = 0; j <= i - 1; j++)
			{
				u = u + X[j][i - 1] * R[i - j];
			}
			K[i] = (R[i] - u) / E[i - 1];
			X[i][i] = K[i];
			for (int j = 1; j <= i - 1; j++)
			{
				X[j][i] = X[j][i - 1] - K[i] * (X[i - j][i - 1]);
			}
			E[i] = (1 - K[i] * K[i])*E[i - 1];
		}
		for (int i = 1; i <= 12; i++)
		{
			W[i] = X[i][12];
			fprintf(frame_file, "%lf\n", W[i]);
		}
		h = h + 80;
		v = 0;
		for (int i = 0; i < 13; i++)
		{
			R[i] = 0;
		}

		//coefficient of cepstral
		C[0] = log(R[0] * R[0]) / log(2.0);
		//calculating the value of c[0];
		for (int m = 1; m <= 12; m++)
		{
			for (int k = 1; k <= m - 1; k++)
			{
				sum = sum + ((long double)k / m)*C[k] * W[m - k];
			}
			C[m] = W[m] + sum;
			sum = 0;
			fprintf(coef_file, "%lf\n", C[m]);
		}

		//cprimes
		for (int m = 1; m <= 12; m++)
		{
			w[m] = 1.0 + 6.0 * sin((22.0 / 7)*m / 12.0);
			//sign raised window is being applied.
			cp[m] = C[m] * w[m];
			fprintf(cptesting, "%Lf\n", cp[m]);
		}
	}
	//printf("before new\n");

	long double min_dis = 999999.0;
	int tokura_index = 0;
	int index_obs[126] = { 0 };
	rewind(codebook_refernce);
	for (int i = 1; i <= 32; i++)
	{
		for (int j = 1; j <= 12; j++)
		{
			fscanf(codebook_refernce, "%lf\n", &codebook[i][j]);
		}
	}
	rewind(cptesting);
	for (int i = 1; i <= 125; i++)
	{
		for (int j = 1; j <= 12; j++)
		{
			fscanf(cptesting, "%lf\n", &all_cprime[i][j]);
			//printf("%lf\n",all_cprime[i][j]);
		}
	}
	for (int i = 1; i <= 125; i++)
	{
		min_dis = 999999.0;
		for (int k = 1; k <= 32; k++)
		{
			toku = 0;
			for (int j = 1; j <= 12; j++)
			{
				toku = toku + (weight[j - 1])*((codebook[k][j] - all_cprime[i][j])*(codebook[k][j] - all_cprime[i][j]));
			}

			if (toku < min_dis)
			{
				min_dis = toku;
				index_obs[i] = k;
			}
		}
	}
	for (int i = 1; i <= 125; i++)
	{
		//printf("OBSERVATION SEQUENCE IS: %d\n", index_obs[i]);
		fprintf(observation_file, "%d\t", index_obs[i]);
		O[i] = index_obs[i];
		//fprintf(testing_observation, "%d\t", index_obs[i]);
	}
	fprintf(observation_file, "\n");
	//fprintf(testing_observation, "\n");

	fclose(open_file);
	fclose(out_file);
	fclose(frame_file);
	fclose(coef_file);
	fclose(rif);
	fclose(cptesting);
	fclose(distance);
	fclose(minimum);
	fclose(codebook_refernce);
	fclose(observation_file);
	//fclose(testing_observation);
}

void return_color_name(long int color_recognized_from_voice)
{
	if (color_recognized_from_voice == 0)
		strcpy(color_recognized_from_voice_name, "ORANGE");
	else if (color_recognized_from_voice == 1)
		strcpy(color_recognized_from_voice_name, "YELLOW");
	else if (color_recognized_from_voice == 2)
		strcpy(color_recognized_from_voice_name, "PINK");
	else if (color_recognized_from_voice == 3)
		strcpy(color_recognized_from_voice_name, "WHITE");
	else if (color_recognized_from_voice == 4)
		strcpy(color_recognized_from_voice_name, "PURPLE");
	else if (color_recognized_from_voice == 5)
		strcpy(color_recognized_from_voice_name, "INDIGO");
	else if (color_recognized_from_voice == 6)
		strcpy(color_recognized_from_voice_name, "GOLDEN");
	else if (color_recognized_from_voice == 7)
		strcpy(color_recognized_from_voice_name, "BLACK");
	else if (color_recognized_from_voice == 8)
		strcpy(color_recognized_from_voice_name, "RED");
	else if (color_recognized_from_voice == 9)
		strcpy(color_recognized_from_voice_name, "BLUE");
	else if (color_recognized_from_voice == 10)
		strcpy(color_recognized_from_voice_name, "GREEN");
	else if (color_recognized_from_voice == 11)
		strcpy(color_recognized_from_voice_name, "BROWN");
	else
		strcpy(color_recognized_from_voice_name, "DAFA HO JAO");

}

void forward()
{
	long double maximum = 0.0;
	int index_digit = 0;

	for (int i = 1; i <= N; i++)
	{
		fscanf(pi_file, "%Le", &pi[i]);
	}

	for (int temp = 0; temp <= 11; temp++)
	{

		FILE *a_file = fopen(name_a[temp], "r");
		FILE *b_file = fopen(name_b[temp], "r");

		for (int i = 1; i <= N; i++)
		{
			for (int j = 1; j <= N; j++)
			{
				fscanf(a_file, "%Le", &a[i][j]);
			}
		}

		for (int i = 1; i <= N; i++)
		{
			for (int j = 1; j <= M; j++)
			{
				fscanf(b_file, "%Le", &b[i][j]);
			}
		}
		//rewind(alpha_open);

		for (int i = 1; i <= N; i++)
		{
			alpha[1][i] = pi[i] * b[i][O[1]];
		}
		for (int t = 1; t <= T - 1; t++)
		{
			for (int j = 1; j <= N; j++)
			{
				sum_alpha = 0.0;
				for (int i = 1; i <= N; i++)
				{
					sum_alpha = sum_alpha + alpha[t][i] * a[i][j];
				}

				alpha[t + 1][j] = sum_alpha*b[j][O[t + 1]];
				//printf("%0.30Lf", alpha[t][j]);
				//fprintf(alpha_open, "%0.30Lf\t", alpha[t][j]);
				//fprintf(alpha_open, "%Le\t", alpha[t][j]);
			}
			//printf("\n");
			//fprintf(alpha_open, "\n");
		}
		prob_alpha[temp] = 0;
		for (int i = 1; i <= N; i++)
		{
			prob_alpha[temp] = prob_alpha[temp] + alpha[T][i];
		}
		//printf("Probability: %Le\n", prob_alpha);
		fclose(a_file);
		fclose(b_file);
	}

	for (int temp = 0; temp <= 11; temp++)
	{
		if (prob_alpha[temp] > maximum)
		{
			maximum = prob_alpha[temp];
			index_digit = temp;
		}
		//printf("\n%Le\n", prob_alpha[temp]);
	}
	printf(" DIGIT IS %d\n", index_digit);
	color_recognized_from_voice = index_digit;
	return_color_name(color_recognized_from_voice);
	fclose(pi_file);

}
































































void print_current_grid()
{
	int i, j, k;
	int x = 10;
	int grid_dimension = 3;
	int size = grid_dimension * x;
	int color = 1;
	int length = 2;

	for (i = 1; i <= size; i++)
	{
		if (i % (x / 2) == 0 && i % x != 0)
		{
			if (color == 1)
			{
				printf("     ORANGE");
				length = strlen("ORANGE");
			}
			else if (color == 2)
			{
				printf("     PINK");
				length = strlen("PINK");
			}
			else if (color == 3)
			{
				printf("     RED");
				length = strlen("RED");
			}
		}

		if (i % (x / 2) == 0 && i % x != 0)
		{
			if (color == 1)
			{
				printf("     \tRED");
				length = strlen("ORANGE");
			}
			else if (color == 2)
			{
				printf("    \tPINK");
				length = strlen("PINK");
			}
			else if (color == 3)
			{
				printf("     \tORANGE");
				length = strlen("RED");
			}
		}



		for (j = 0; j < size; j++)
		{
			if (i == 1)
				printf("-");
			else if (j % x == 0 && i % (x / 2) == 0 && i % x != 0)
			{
				printf("");
			}
			else if (j % x == 0)
			{
				printf("`");
			}
			else
			{
				printf(" ");
			}
			if (i % x == 0)
			{
				printf("\b-");
			}
			if (i % (x / 2) == 0 && i % x != 0)
			{
				//printf("\b");
			}
		}
		color = 3;
		if (!(i % (x / 2) == 0 && i % x != 0))
		{
			//printf("\b");
			printf("|\n");
			for (k = 0; k < length; k++){}
			//printf("\b");
		}

	}

}

void print_grid(long int current_size)
{
	long int i, j;
	long int row = 0;
	long int column = 0;

	printf("\n");
	if (current_size == 4)
	{
		row = 2;
		column = 2;
		for (i = 0; i < row; i++)
		{
			for (j = 0; j < column; j++)
			{
				if (id_array[i][j] == 0)
				{
					printf("\t\t%s", color_array[i][j]);
				}
				else
				{
					printf("\t\t%s", image_array[i][j]);
				}
			}
			printf("\n\n");
		}

	}
	if (current_size == 6)
	{
		row = 2;
		column = 3;
		for (i = 0; i < row; i++)
		{
			for (j = 0; j < column; j++)
			{
				if (id_array[i][j] == 0)
				{
					printf("\t\t%s", color_array[i][j]);
				}
				else
				{
					printf("\t\t%s", image_array[i][j]);
				}
			}
			printf("\n\n");
		}

	}
	if (current_size == 8)
	{
		row = 2;
		column = 4;
		for (i = 0; i < row; i++)
		{
			for (j = 0; j < column; j++)
			{
				if (id_array[i][j] == 0)
				{
					printf("\t\t%s", color_array[i][j]);
				}
				else
				{
					printf("\t\t%s", image_array[i][j]);
				}
			}
			printf("\n\n");
		}
	}




}

void initialize_id_array(long int current_size)
{
	long int i, j;
	long int row = 0;
	long int column = 0;

	if (current_size == 4)
	{
		row = 2;
		column = 2;
		for (i = 0; i < row; i++)
		{
			for (j = 0; j < column; j++)
			{
				id_array[i][j] = 0;
			}
		}
	}
	else if (current_size == 6)
	{
		row = 2;
		column = 3;
		for (i = 0; i < row; i++)
		{
			for (j = 0; j < column; j++)
			{
				id_array[i][j] = 0;
			}
		}
	}
	else if (current_size == 8)
	{
		row = 2;
		column = 4;
		for (i = 0; i < row; i++)
		{
			for (j = 0; j < column; j++)
			{
				id_array[i][j] = 0;
			}
		}
	}



}

void initialize_image_color_array(long int current_size)
{

	long int i, j;
	long int index = 0;
	long int row = 0;
	long int column = 0;

	for (i = 0; i < current_size; i++)
	{
		check_status[i] = 0;
	}

	if (current_size == 4)
	{
		row = 2;
		column = 2;

		for (i = 0; i < row; i++)
		{
			for (j = 0; j < column; j++)
			{
				strcpy(color_array[i][j], sample_color[index++]);
				if ((i == 0 && j == 0) || (i == 1 && j == 1))
				{
					strcpy(image_array[i][j], sample_image[0]);
				}
				else
				{
					strcpy(image_array[i][j], sample_image[1]);
				}
			}
		}

	}

	else if (current_size == 6)
	{
		row = 2;
		column = 3;
		for (i = 0; i < row; i++)
		{
			for (j = 0; j < column; j++)
			{
				strcpy(color_array[i][j], sample_color[index++]);
				if ((i == 0 && j == 0) || (i == 0 && j == 2))
				{
					strcpy(image_array[i][j], sample_image[0]);
				}
				else if ((i == 1 && j == 0) || (i == 1 && j == 1))
				{
					strcpy(image_array[i][j], sample_image[1]);
				}
				else if ((i == 0 && j == 1) || (i == 1 && j == 2))
				{
					strcpy(image_array[i][j], sample_image[2]);
				}
			}
		}
	}

	else if (current_size == 8)
	{
		row = 2;
		column = 4;
		for (i = 0; i < row; i++)
		{
			for (j = 0; j < column; j++)
			{
				strcpy(color_array[i][j], sample_color[index++]);

				if ((i == 0 && j == 0) || (i == 1 && j == 3))
				{
					strcpy(image_array[i][j], sample_image[1]);
				}
				else if ((i == 0 && j == 1) || (i == 0 && j == 3))
				{
					strcpy(image_array[i][j], sample_image[2]);
				}
				else if ((i == 0 && j == 2) || (i == 1 && j == 2))
				{
					strcpy(image_array[i][j], sample_image[3]);
				}
				else if ((i == 1 && j == 0) || (i == 1 && j == 1))
				{
					strcpy(image_array[i][j], sample_image[4]);
				}
			}
		}
	}
}

long int return_id(char input_color[], long int current_size)
{
	if (current_size == 4)
	{
		if (strcmp(input_color, "ORANGE") == 0)
			return 0;
		if (strcmp(input_color, "YELLOW") == 0)
			return 1;
		if (strcmp(input_color, "PINK") == 0)
			return 2;
		if (strcmp(input_color, "WHITE") == 0)
			return 3;
		else
			return INFINITY;
	}

	if (current_size == 6)
	{
		if (strcmp(input_color, "ORANGE") == 0)
			return 0;
		if (strcmp(input_color, "YELLOW") == 0)
			return 1;
		if (strcmp(input_color, "PINK") == 0)
			return 2;
		if (strcmp(input_color, "WHITE") == 0)
			return 3;
		if (strcmp(input_color, "PURPLE") == 0)
			return 4;
		if (strcmp(input_color, "INDIGO") == 0)
			return 5;
		else
			return INFINITY;
	}

	if (current_size == 8)
	{
		if (strcmp(input_color, "ORANGE") == 0)
			return 0;
		if (strcmp(input_color, "YELLOW") == 0)
			return 1;
		if (strcmp(input_color, "PINK") == 0)
			return 2;
		if (strcmp(input_color, "WHITE") == 0)
			return 3;
		if (strcmp(input_color, "PURPLE") == 0)
			return 4;
		if (strcmp(input_color, "INDIGO") == 0)
			return 5;
		if (strcmp(input_color, "GOLDEN") == 0)
			return 6;
		if (strcmp(input_color, "BLACK") == 0)
			return 7;
		else
			return INFINITY;
	}

}

void return_coordinate(long int id, long int current_size, long int *x, long int *y)
{
	long int row, column;

	if (current_size == 4)
	{
		row = 2;
		column = 2;
		*x = id / column;
		*y = id % column;

	}
	else if (current_size == 6)
	{
		row = 2;
		column = 3;
		*x = id / column;
		*y = id % column;

	}
	else if (current_size == 8)
	{
		row = 2;
		column = 4;
		*x = id / column;
		*y = id % column;

	}
}

void if_matched(long int current_size)
{

	printf("\nPattern Matched.....\n");
	Sleep(1000);
	return_coordinate(id2, current_size, &x, &y);
	id_array[x][y] = 1;
	check_status[id2] = 1;
	check_status[id1] = 1;
	answer = answer + 2;
	score += 5;
}

void if_not_matched(long int current_size)
{
	if (id2 == INFINITY)
	{
		printf("\nEntr Valid choice\n");
		return_coordinate(id1, current_size, &x, &y);
		id_array[x][y] = 0;
		Sleep(1000);
		return;
	}
	check_status[id1] = 0;
	return_coordinate(id2, current_size, &x, &y);
	id_array[x][y] = 1;
	printf("\nNot matched , Try again\n");
	print_grid(current_size);
	if (check_status[id2] != 1)
		id_array[x][y] = 0;
	return_coordinate(id1, current_size, &x, &y);
	id_array[x][y] = 0;
	score -= 2;
	Sleep(3000);

}

void play_game(long int current_size)
{

	if (current_size == 4)
	{
		answer = 0;
		id1 = 0; id2 = 0;
		score = 0;
		while (answer != 4)
		{

			system("cls");
			printf("LEVEL 1:\n");
			print_grid(current_size);
			printf("Enter your 1st color choice");

			//scanf("%s", input_color);
			detect_color(current_size);


			strcpy(input_color, color_recognized_from_voice_name);
			printf("The color spoken is %s", input_color);
			Sleep(2000);
			id1 = return_id(input_color, current_size);
			if (id1 < 4 && check_status[id1] == 0)
			{
				return_coordinate(id1, current_size, &x, &y);
				id_array[x][y] = 1;
				//check_status[id1] = 1;
			}
			/*			else if(id1 < 8)
			{
			printf("Enter valid choice\n");
			Sleep(1000);
			return_coordinate(id1, current_size, &x, &y);
			id_array[x][y] = 0;
			continue;
			}*/
			else
			{
				printf("Invalid Choice\n");
				Sleep(1000);
				continue;
			}
			system("cls");
			printf("The updated Grid is:\n");
			print_grid(current_size);
			printf("\nEnter your 2nd color choice\n");

			//scanf("%s", input_color);
			detect_color(current_size);
			strcpy(input_color, color_recognized_from_voice_name);
			printf("The color spoken is %s", input_color);
			Sleep(2000);
			id2 = return_id(input_color, current_size);

			if (id1 == 0)
			{
				if (id2 == 3)
				{
					if_matched(current_size);
				}
				else
				{
					//printf("I am here\n");
					if_not_matched(current_size);

					continue;
				}
			}
			if (id1 == 1)
			{
				if (id2 == 2)
				{
					if_matched(current_size);
				}
				else
				{
					if_not_matched(current_size);
					continue;
				}
			}
			if (id1 == 2)
			{
				if (id2 == 1)
				{
					if_matched(current_size);
				}
				else
				{
					if_not_matched(current_size);
					continue;
				}

			}
			if (id1 == 3)
			{
				if (id2 == 0)
				{
					if_matched(current_size);
				}
				else
				{
					if_not_matched(current_size);
					continue;
				}
			}
		}
		overall_score += score;
		printf("congratulation you have won the game with a score of: %ld \n", score);
		print_grid(current_size);
		Sleep(3000);
	}


	if (current_size == 6)
	{
		answer = 0;
		id1 = 0; id2 = 0;
		score = 0;
		while (answer != 6)
		{

			system("cls");
			printf("LEVEL 2:\n");
			print_grid(current_size);
			printf("\nEnter your 1st color choice\n");

			//scanf("%s", input_color);
			detect_color(current_size);
			strcpy(input_color, color_recognized_from_voice_name);
			printf("The color spoken is %s\n", input_color);
			Sleep(2000);
			id1 = return_id(input_color, current_size);
			if (id1 < 6 && check_status[id1] == 0)
			{
				return_coordinate(id1, current_size, &x, &y);
				//check_status[id1] = 1;
				id_array[x][y] = 1;
			}
			else
			{
				printf("Enter valid choice\n");
				continue;
			}
			system("cls");
			printf("The updated Grid is:\n");
			print_grid(current_size);
			printf("\nEnter your 2nd color choice\n");

			//scanf("%s", input_color);
			detect_color(current_size);
			strcpy(input_color, color_recognized_from_voice_name);
			printf("The color spoken is %s\n", input_color);
			Sleep(2000);
			id2 = return_id(input_color, current_size);
			if (id1 == 0)
			{
				if (id2 == 2)
				{
					if_matched(current_size);
				}
				else
				{
					if_not_matched(current_size);
					continue;
				}
			}
			if (id1 == 1)
			{
				if (id2 == 5)
				{
					if_matched(current_size);
				}
				else
				{
					if_not_matched(current_size);
					continue;
				}
			}
			if (id1 == 2)
			{
				if (id2 == 0)
				{
					if_matched(current_size);
				}
				else
				{
					if_not_matched(current_size);
					continue;
				}

			}
			if (id1 == 3)
			{
				if (id2 == 4)
				{
					if_matched(current_size);
				}
				else
				{
					if_not_matched(current_size);
					continue;
				}
			}
			if (id1 == 4)
			{
				if (id2 == 3)
				{
					if_matched(current_size);
				}
				else
				{
					if_not_matched(current_size);
					continue;
				}
			}
			if (id1 == 5)
			{
				if (id2 == 1)
				{
					if_matched(current_size);
				}
				else
				{
					if_not_matched(current_size);
					continue;
				}
			}
		}
		overall_score += score;
		printf("Congratulation you have won the game with a score of: %ld \n", score);
		print_grid(current_size);
		Sleep(3000);
	}


	if (current_size == 8)
	{
		answer = 0;
		id1 = 0; id2 = 0;
		score = 0;
		while (answer != 8)
		{

			system("cls");
			printf("LEVEL 3:\n");
			print_grid(current_size);
			printf("\nEnter your 1st color choice\n");

			//scanf("%s", input_color);
			detect_color(current_size);
			strcpy(input_color, color_recognized_from_voice_name);
			printf("The color spoken is %s\n", input_color);
			Sleep(2000);
			id1 = return_id(input_color, current_size);
			if (id1 < 8 && check_status[id1] == 0)
			{
				return_coordinate(id1, current_size, &x, &y);
				id_array[x][y] = 1;
				//check_status[id1] = 1;
			}
			else
			{
				printf("Enter valid choice\n");
				continue;
			}
			system("cls");
			printf("The updated Grid is:\n");
			print_grid(current_size);
			printf("\nEnter your 2nd color choice\n");

			//scanf("%s", input_color);
			detect_color(current_size);
			strcpy(input_color, color_recognized_from_voice_name);
			printf("The color spoken is %s", input_color);
			Sleep(2000);
			id2 = return_id(input_color, current_size);
			if (id1 == 0)
			{
				if (id2 == 7)
				{
					if_matched(current_size);
				}
				else
				{
					if_not_matched(current_size);
					continue;
				}
			}
			if (id1 == 1)
			{
				if (id2 == 3)
				{
					if_matched(current_size);
				}
				else
				{
					if_not_matched(current_size);
					continue;
				}
			}
			if (id1 == 2)
			{
				if (id2 == 6)
				{
					if_matched(current_size);
				}
				else
				{
					if_not_matched(current_size);
					continue;
				}

			}
			if (id1 == 3)
			{
				if (id2 == 1)
				{
					if_matched(current_size);
				}
				else
				{
					if_not_matched(current_size);
					continue;
				}
			}
			if (id1 == 4)
			{
				if (id2 == 5)
				{
					if_matched(current_size);
				}
				else
				{
					if_not_matched(current_size);
					continue;
				}
			}
			if (id1 == 5)
			{
				if (id2 == 4)
				{
					if_matched(current_size);
				}
				else
				{
					if_not_matched(current_size);
					continue;
				}
			}
			if (id1 == 6)
			{
				if (id2 == 2)
				{
					if_matched(current_size);
				}
				else
				{
					if_not_matched(current_size);
					continue;
				}
			}
			if (id1 == 7)
			{
				if (id2 == 0)
				{
					if_matched(current_size);
				}
				else
				{
					if_not_matched(current_size);
					continue;
				}
			}
		}
		overall_score += score;
		printf("congratulation you have won the game with a score of: %ld \n", score);
		print_grid(current_size);
		Sleep(3000);
	}
}

void detect_color(long int current_size)
{
	/*system("Recording_Module.exe 3 silence.wav silence.txt");
	system("Recording_Module.exe 3 input.wav input.txt");

	dcshift("silence.txt", "input.txt", "output_DC.txt");
	normalisation("output_DC.txt", "output_normal.txt");
	triming("output_normal.txt", "trim.txt");
	recognition();
	forward();
	//observation_division();
	fclose(alpha_open);
	*/
	long long int no_of_samples = 0;
	double max_value = 0.0;
	double ratio = 0.0;
	long long int ret = 0;
	double DC_SHIFT_VALUE = 0.0;
	double normalized_value = NORMALIZE_VALUE;
	char enter;
	double arr[100000];
	double sample_array[20000] = { 0 };
	double hamming_window[321] = { 0 };
	double frame_array[321] = { 0 };
	long long int start = 0, end = 0;
	long double arr_Ri[30] = { 0 };
	long double arr_Ai[30] = { 0 };
	long double arr_Ci[30] = { 0 };
	long double arr_Cdash[30] = { 0 };
	long long int sliding_index = 0;
	long long int length = 0;
	long long int no_of_frames = NO_OF_OBSERVATION, i = 0, j = 0;
	long double find_vowel_distance[6] = { 0.0 };
	char reference_file_name[300];
	long long int minimum_index = 0;
	long double minimum = 999999.0;
	char wait;
	long int digit = 0;
	long double max_probability = 0;
	long int digit_spoken = 0;
	char intermediate_file_index[3];
	int correctness = 0;
	int user_choice;
	probability_of_O_given_model = 0;

	read_codebook();
	char input_test_name[400];
	char intermediate[300];
	//	scanf("%s", intermediate);

	/* Taking Silence file Name */
	printf("\n\t Please maintain silence for 3 seconds:\n");
	system("Recording_Module.exe 3 silence_file.wav silence_file.txt");

	/* Taking input file name */
	printf("\n\t Please speak visible colors \n");
	system("Recording_Module.exe 3 input_file.wav input_file.txt");


	//	strcpy(input_test_name, "Input/");
	//	strcat(input_test_name, intermediate);
	silence_fp = fopen("silence_file.txt", "r+");
	//input_fp =		fopen(input_test_name, "r+");
	input_fp = fopen("input_file.txt", "r+");
	operation_fp = fopen("sample_output.txt", "w+");
	dump_Ri = fopen("dump_Ri_test.txt", "w+");
	dump_Ai = fopen("dump_Ai_test.txt", "w+");
	dump_Ci = fopen("dump_Ci_test.txt", "w+");
	dump_Cdash = fopen("dump_Cdash_test.txt", "w+");
	sample_file_fp = fopen("Sample_file.txt", "w+");
	hamming_file = fopen(HAMMING_WINDOW, "r+");


	DC_SHIFT_VALUE = find_DC_shift(silence_fp);
	no_of_samples = count_no_of_sample(input_fp);
	max_value = find_max(input_fp);
	copy_to_array(input_fp, arr, &length);
	//printf("No of saMples in whole file are : %lld\n max is %Lf\n", length, max_value);
	find_frames(sample_array, arr, &start, &end, length, sample_file_fp);
	get_hamming_window(hamming_file, hamming_window);

	fclose(input_fp);
	fclose(sample_file_fp);

	input_fp = fopen("Sample_file.txt", "r+");


	ratio = (double)normalized_value / max_value;
	//printf("normalized value is %Lf\n", (long double)normalized_value / max_value);
	ret = apply_DC_shift_Normalize(input_fp, operation_fp, DC_SHIFT_VALUE, ratio);
	no_of_samples = count_no_of_sample(input_fp);
	//	printf("DC Shift obtained is : %lf\n", DC_SHIFT_VALUE);
	//	printf("No of samples : %lld\n", no_of_samples);
	//	printf("Max value amoung samples: %lf\n", max_value);

	fclose(operation_fp);
	operation_fp = fopen("sample_output.txt", "r+");
	copy_to_array(operation_fp, sample_array, &length);


	for (i = 0; i < FRAME_SIZE; i++)
	{
		frame_array[i] = sample_array[i];
	}

	for (i = 0; i < no_of_frames; i++)
	{
		apply_hamming_window(hamming_window, frame_array, sample_array, sliding_index + i * 80);
		do_autocorelation(frame_array, arr_Ri, length, dump_Ri);
		do_durbin(arr_Ri, dump_Ai, arr_Ai);
		do_capstral(arr_Ai, arr_Ri, arr_Ci, dump_Ci);
		apply_sine_window(arr_Cdash, arr_Ci, dump_Cdash);
	}

	fclose(dump_Cdash);
	dump_Cdash = fopen("dump_Cdash_test.txt", "r+");

	find_observation_sequence();
	print_Observation_Sequence();


	for (digit = 0; digit < current_size; digit++)
	{
		if (check_status[digit] == 1)
		{
			//printf("////////////////////////////////////////////////////////");
			continue;
		}

		strcpy(HMM_AIJ, "Input/a_");
		sprintf(intermediate_file_index, "%ld", digit);
		strcat(HMM_AIJ, intermediate_file_index);
		strcat(HMM_AIJ, ".txt");

		strcpy(HMM_BJK, "Input/b_");
		sprintf(intermediate_file_index, "%ld", digit);
		strcat(HMM_BJK, intermediate_file_index);
		strcat(HMM_BJK, ".txt");

		strcpy(HMM_PII, "Input/pi.txt");
		//sprintf(intermediate_file_index, "%ld", digit);
		//strcat(HMM_PII, intermediate_file_index);
		//strcat(HMM_PII, ".txt");


		read_Model();
		//printf("hello\n");
		/*  SOLUTION 1 */
		/* Forward Procedure */
		forward_step1_initialization();
		forward_step2_induction();
		forward_step3_termination();
		//printf("P(O|lambda): for digit %ld %0.30Le \n\n", digit, probability_of_O_given_model);
		if (probability_of_O_given_model >= max_probability)
		{
			max_probability = probability_of_O_given_model;
			digit_spoken = digit;
		}

	}
	//printf("The digit recognized is %ld\n", digit_spoken);
	//printf(".......................................................... %d ", digit_spoken);
	color_recognized_from_voice = digit_spoken;
	return_color_name(color_recognized_from_voice);
	fclose(silence_fp);
	fclose(input_fp);
	fclose(operation_fp);

	fclose(dump_Ri);
	fclose(dump_Ai);
	fclose(dump_Ci);
	fclose(dump_Cdash);
	//fclose(sample_file_fp);
	fclose(hamming_file);
	//printf("Reach here\n");

}

int _tmain(int argc, _TCHAR* argv[])
{
	char wait;
	long int level;
	level_fp = fopen(LEVEL_FILE, "r+");
	int current_size = 2;


	fscanf(level_fp, "%ld", &level);
	//printf("Current level is : %ld\n", level);
	Sleep(3000);
	printf("\n\n\n");
	if (level == 1)
	{
		system("cls");
		printf("\n\n\n\n\n\n\n\t\t\t\t\tEnter in level 1... \n\n");
		current_size = 4;
		Sleep(3000);
		initialize_image_color_array(current_size);
		initialize_id_array(current_size);
		print_grid(current_size);
		play_game(current_size);
		PlaySound("play1.wav", NULL, SND_FILENAME); //SND_FILENAME or SND_LOOP
		level = 2;

	}

	if (level == 2)
	{
		system("cls");
		printf("\n\n\n\n\n\n\n\n\n\t\t\t\t\t\Enter in level 2... \n\n");
		Sleep(3000);
		current_size = 6;
		initialize_image_color_array(current_size);
		initialize_id_array(current_size);
		print_grid(current_size);
		play_game(current_size);
		PlaySound("play1.wav", NULL, SND_FILENAME); //SND_FILENAME or SND_LOOP
		level = 3;
	}
	printf("\n\n");
	if (level == 3)
	{
		system("cls");
		printf("\n\n\n\n\n\n\n\n\t\t\t\t\t\Enter in level 3... \n\n");
		Sleep(3000);
		current_size = 8;
		initialize_image_color_array(current_size);
		initialize_id_array(current_size);
		print_grid(current_size);
		play_game(current_size);
		
	}
	system("cls");
	printf("\n\n\n\n\n\n\n\n\n\t\tGame completed, The overall score is: %d out of 45\n", overall_score);
	PlaySound("play1.wav", NULL, SND_FILENAME); //SND_FILENAME or SND_LOOP

exit:
	//print_current_grid();
	scanf("%c", &wait);
	scanf("%c", &wait);
	return 0;
}

