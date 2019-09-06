// HMM_Solution_3.cpp : Defines the entry point for the console application.
// Application to find solution of Problem 2 and Problem 3
/* Version 1.0 */
/* @author Shubham Jain 
	24-10-18
*/

/* What to do in short?
1. Find forward and backward procedure to be used in Solution 3.
2. Use viterbi to find the state sequence and p*.
3. Use EM(expectation-maximization) to update model.
4. Loop between step 1 to step 3 till we get a good trained model.
*/

/* Includes */
#include "stdafx.h"
#include "stdio.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
#include <windows.h>

/* Defines */
#define MAJOR_NUMBER				1
#define MINOR_NUMBER				2
#define VECTOR_SIZE					12
#define MAX_FILE_NAME_SIZE			200

#define MAJOR_NUMBER				1
#define MINOR_NUMBER				2
#define VECTOR_SIZE					12
#define NO_OF_COLOR					8
#define NO_OF_OBSERVATION		125
#define NO_OF_SAMPLE			320 + (NO_OF_OBSERVATION - 1) * 80
#define INPUT_CEPSTRAL			"dump_Cdash_test.txt"
#define INT_MAX					999999
#define SILENCE_FILE				"Input/silence_file.txt"
#define FRAME_SIZE				320
#define NORMALIZE_VALUE			5000

int TRIM_WINDOW_SHIFT = 250;
int INTERVAL_SIZE = 320 + (NO_OF_OBSERVATION - 1) * 80;

/* File to be used in the Implementation */
//#define HMM_AIJ						"HMM_AIJ.txt"
//#define HMM_BJK						"HMM_BJK.txt"
//#define HMM_AIJ						"7a.txt"
//#define HMM_BJK						"7b.txt"
//#define HMM_PII						"HMM_PII.txt"
//#define HMM_AIJ_FINAL				"HMM_AIJ_FINAL.txt"
//#define HMM_BJK_FINAL				"HMM_BJK_FINAL.txt"
//#define HMM_PII_FINAL				"HMM_PII_FINAL.txt"
//#define HMM_OBSERVATION_SEQUENCE	"HMM_OBSERVATION_SEQUENCE.txt"
#define HMM_ALPHA_MATRIX			"HMM_ALPHA_MATRIX.txt"
#define HMM_BETA_MATRIX				"HMM_BETA_MATRIX.txt"
#define HMM_STATE_SEQUENCE			"HMM_STATE_SEQUENCE_VITERBI.txt"
#define CODEBOOK_FILE_NAME			"Input/codebook.txt"
#define HAMMING_WINDOW				"Input/Hamming_window.txt"
#define COLOR_NAME_SIZE					100

/* Debug print Enable/Disable */
#define DEBUG_PRINT
#define DEBUG_AIJ
#define DEBUG_BJK
//#define DEBUG_PII
//#define DEBUG_ALPHA
//#define DEBUG_BETA
//#define DEBUG_DELTA
//#define DEBUG_PSI
//#define DEBUG_XII
//#define DEBUG_GAMMA
#define CODEBOOK_SIZE			32

/* File to be used in the Implementation */
//#define HMM_AIJ						"HMM_AIJ.txt"
//#define HMM_BJK						"HMM_BJK.txt"
//#define HMM_PII						"HMM_PII.txt"
//#define HMM_AIJ_FINAL				"HMM_AIJ_FINAL.txt"
//#define HMM_BJK_FINAL				"HMM_BJK_FINAL.txt"
//#define HMM_PII_FINAL				"HMM_PII_FINAL.txt"
#define HMM_ALPHA_MATRIX			"HMM_ALPHA_MATRIX.txt"
#define HMM_BETA_MATRIX				"HMM_BETA_MATRIX.txt"
#define HMM_STATE_SEQUENCE			"HMM_STATE_SEQUENCE_VITERBI.txt"
char sample_color[8][100] =	 {"ORANGE", "YELLOW", "PINK" , "WHITE", "PURPLE", "INDIGO", "GOLDEN", "BLACK"};
#define HAMMING_WINDOW				"Input/Hamming_window.txt"

/* Debug print Enable/Disable */
//#define DEBUG_PRINT
//#define DEBUG_AIJ
//#define DEBUG_BJK
//#define DEBUG_PII
//#define DEBUG_ALPHA
//#define DEBUG_BETA
//#define DEBUG_DELTA
//#define DEBUG_PSI
//#define DEBUG_XII
//#define DEBUG_GAMMA
//#define B_SUMMATION_CHECK
FILE *obs_seq_open ;

FILE *codebook_fp, *input_fp_cepstral;
FILE *input_fp, *operation_fp, *dump_Ri, *dump_Ai, *dump_Ci, *silence_fp, *sample_file_fp, *hamming_file, *dump_Cdash;
FILE *dump_Cdash_reference, *deleteit;
long double codebook[CODEBOOK_SIZE][VECTOR_SIZE] ={0};
long double array_input_cepstral[NO_OF_OBSERVATION][VECTOR_SIZE] = {0};







FILE *output_fp_alpha, *output_fp_beta, *output_fp_gamma, *output_state_sequence, *output_state_sequence_gamma;
FILE *output_fp_aij, *output_fp_bjk, *output_fp_pii;
FILE *input_fp_aij, *input_fp_bjk, *input_fp_pii, *input_fp_observation_sequence, *input_fp_alpha, *input_fp_beta;

//#define DEBUG_OBSERVATION_SEQUENCE_PRINT
//#define DEBUG_STATE_SEQUENCE_PRINT

/* No of states in the model (N) */
#define NO_OF_STATE					5	

/* No of distict observation symbols per state (M) */
#define NO_OF_OBSERVATION_SYMBOL	32				

/* No of observation for a given observation sequence (T) */
#define NO_OF_OBSERVATION			125			

/* Threshold value to update Bjk matrix */
#define BJK_THRESHOLD				pow(10.0, -30.0)

/* Deciding which gamma to use, Solution 2 or Solution 3, though both are same. Solution 3 gamma do not a last row */
#define OLD_GAMMA_SOLUTION2

/* Number of iteration to use for training the model */
#define NO_OF_ITERATION_TO_TRAIN	20

/* Threshold value for Pstar */
#define THRESHOLD_PSTAR				pow(10.0, -40)

//#define DIGIT						0
//#define DIGIT_UTTERANCE_COUNT		1
//#define NO_OF_TIMES_TO_AVERAGE		1

long int DIGIT					 = 7;
long int DIGIT_UTTERANCE_COUNT   = 15;
long int NO_OF_TIMES_TO_AVERAGE  = 3;

/* Globals */
long double array_alpha_ti[NO_OF_STATE + 1][NO_OF_OBSERVATION + 1] = {0};
long double array_beta_ti[NO_OF_STATE + 1][NO_OF_OBSERVATION + 1] = {0};
long double array_aij[NO_OF_STATE + 1][NO_OF_STATE + 1] = {0};
long double array_bjk[NO_OF_STATE + 1][NO_OF_OBSERVATION_SYMBOL + 1] = {0};
long double array_pii[NO_OF_STATE + 1] = {0};

long double array_aij_copy[NO_OF_STATE + 1][NO_OF_STATE + 1] = {0};
long double array_bjk_copy[NO_OF_STATE + 1][NO_OF_OBSERVATION_SYMBOL + 1] = {0};
long double array_pii_copy[NO_OF_STATE + 1] = {0};

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

long double array_delta_ti[NO_OF_STATE + 1][NO_OF_OBSERVATION + 1] = {0};
long double array_XI_ijt[NO_OF_STATE + 1][NO_OF_STATE + 1][NO_OF_OBSERVATION + 1] = {0};
long double array_gamma_ti[NO_OF_STATE + 1][NO_OF_OBSERVATION + 1] = {0};
long double array_summation[NO_OF_OBSERVATION + 1] = {0};
long int array_psi_ti[NO_OF_STATE + 1][NO_OF_OBSERVATION + 1] = {0};
long int array_state_sequence[NO_OF_OBSERVATION + 1] = {0};
long int array_observation_sequence[NO_OF_OBSERVATION + 1] = {0};
long double b_summation[NO_OF_STATE + 1] = {0};
long double Pstar = 0.0;
long int QstarT = 0; 
long double probability_of_O_given_model = 0.0;
long double last_Pstar = pow(10.0, -160);
long int flag = 0;
long int check = 0;

char model_matrix_A_name[MAX_FILE_NAME_SIZE];
char model_matrix_B_name[MAX_FILE_NAME_SIZE];
char model_matrix_Pi_name[MAX_FILE_NAME_SIZE];
char HMM_AIJ_FINAL[MAX_FILE_NAME_SIZE];
char HMM_BJK_FINAL[MAX_FILE_NAME_SIZE];
char HMM_PII_FINAL[MAX_FILE_NAME_SIZE];

char HMM_AVERAGE_AIJ[MAX_FILE_NAME_SIZE];
char HMM_AVERAGE_BJK[MAX_FILE_NAME_SIZE];
char HMM_AVERAGE_PII[MAX_FILE_NAME_SIZE];

char HMM_AIJ[MAX_FILE_NAME_SIZE];
char HMM_BJK[MAX_FILE_NAME_SIZE];
char HMM_PII[MAX_FILE_NAME_SIZE];




FILE *output_fp_average_aij, *output_fp_average_bjk, *output_fp_average_pii;



char HMM_OBSERVATION_SEQUENCE[MAX_FILE_NAME_SIZE] = {0};

/* Function Declaration */
void print_info();
void read_Model();
void forward_step1_initialization();
void forward_step2_induction();
void forward_step3_termination();
void backward_step1_initialization();
void backward_step2_induction();
void calculate_gamma();
void viterbi_step1_initialization();
void viterbi_step2_recursion();
void viterbi_step3_termination();
void viterbi_step4_path();
void print_Observation_Sequence();
void print_State_Sequence_Pstar();
void calculate_Xi();
void calculate_PII();
void calculate_AIJ();
void calculate_BJK();
void threshold_to_Bjk();
void initialize_all_matrices();
void update_model_to_file();
void debug_print();
void read_Alpha_Beta();
void b_summation_check();


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
	while(1)
	{
		if(feof(silence_input_fp))
		{
			break;
		}
		count++;
		fscanf(silence_input_fp, "%Lf", &value);
		
		/* Discarding first 200 sample */
		if(count > 200)
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
	
	while(1)
	{
		if(feof(input_fp))
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
	
	while(1)
	{
		if(feof(input_fp))
		{
			break;
		}
		fscanf(input_fp, "%lf", &value);
		if(max < abs(value))
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
	
	while(1)
	{
		if(feof(input_fp))
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

	for(i = 0 ;i < 13; i++)
	{
		arr_Ri[i] = 0;
	}

	frame_count = length / FRAME_SIZE;

		for(i = 0; i <= p; i++)
		{
			for(j = 0; j < FRAME_SIZE - i ; j++)
			{
				arr_Ri[i] = arr[j]*arr[j+i] + arr_Ri[i];
			}
		}

	for(i = 0; i <= p ; i++)
	{
		//printf("R %lld is : %Lf\n", i, arr_Ri[i]);
		fprintf(dump_Ri, "%Lf\t",arr_Ri[i]);
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
	for(i = 0 ;i < 13; i++)
	{
		arr_Ai[i] = 0;
	}

	long double Energy[20]    = {0};
	long double alpha[20][20] = {0};
	long long int j = 0;
	long double coefficient_K[20]= {0};
	long long int p = VECTOR_SIZE;

	if(arr_Ri[0] < 60)
	{
		//printf("Error: Energy cannot be so less\n");
		return;
	}

	/* Assigning R[0] to E[0] */
	Energy[0] = arr_Ri[0];

	for( i = 1; i <= p; i++)
	{
		if( i == 1)
		{
			coefficient_K[i] = arr_Ri[1] / arr_Ri[0];
			alpha[i][i] = coefficient_K[i];
		}
		else
		{
			for(j = 1; j <= i-1; j++)
			{
				coefficient_K[i] = alpha[j][i-1] * arr_Ri[i-j] + coefficient_K[i]; 
			}
			coefficient_K[i] = (arr_Ri[i] - coefficient_K[i]) / Energy[i-1];
			alpha[i][i] = coefficient_K[i];

			if(i > 1)
			{
				for(j = 1; j <= i-1; j++)
				{
					alpha[j][i] = alpha[j][i-1] - coefficient_K[i] * alpha[i-j][i-1];		
				}
			}

		}
		Energy[i] = (1 - coefficient_K[i] * coefficient_K[i]) * Energy[i-1];
	}
	for(j = 0; j <= p; j++)
	{
		arr_Ai[j] = alpha[j][12];
	}

	for(j = 1 ; j<=12 ; j++)
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

	for(i = 0 ; i < 14; i++)
	{
		arr_Ci[i] = 0;
	}

	arr_Ci[0] = 2.0 * log(arr_Ri[0]) / log(2.0);
	//printf("gain factor :%lf\n\n", arr_Ci[0]);
	for(m = 1; m <= p ; m++)
	{
		x = 0;
		for(k = 1; k <= m - 1 ; k++)
		{
			x = ((double)k / m) * arr_Ci[k] * arr_Ai[m-k] + x; 
		}
		arr_Ci[m] = arr_Ai[m] + x;
	}

	for(i = 1; i <= 12 ; i++)
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
	long double window[14] = {0};
	for(m = 1; m <= p; m++)
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
		return ;
	}
	rewind(operation_fp);
	
	while(1)
	{
		if(feof(operation_fp))
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

	long long int i = 0,j;
	long long int index = 0;
	double max_value = 0;
	long long int x,y;
	long double value = 0.0;
	long long int count = 0;
	

	for(i = 0; i < length; i++)
	{
		if(max_value < abs(arr[i]))
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

	if(length < 7040)
	{
		printf("The number of sample in files are very less");
	}

	if(x < 0)
	{
		x = 1;
		y = INTERVAL_SIZE;
	}
	if(y >= length)
	{
		y = length - 2;
		x = y - INTERVAL_SIZE + 1;
	}
	
	for(i = x; i < y; i++)
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

	long long int i = 0,j;
	long long int index = 0;
	double max_value = 0;
	long long int x = 0,y = 0;;
	long double value = 0.0;
	int count = 0;
	long double energy = 0;
	long double intermediate_enery = 0;
	int start_marker = 0;
	int end_marker = 0;
	//int TRIM_WINDOW_SHIFT1 = 250;
	//int INTERVAL_SIZE = 9840;

	length = length - 1;
	//printf("The length of input obtained is %ld\n", length);
	for(i = 0 ; i < length - 1;i++)
	{
		count++;
		//printf("value of iiiiii is .......................................... %d .......\n", i);
		intermediate_enery += 0.01 * arr[i] * arr[i];
		if(count == INTERVAL_SIZE)
		{
			//printf("count ::%ld\n ",count);
			count = 0;
			if(energy < intermediate_enery)
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

	//printf(" Start marker is %d \n", start_marker);
	//printf("End marker is %d\n", end_marker);

	if(x > 1000)
	{
		x = start_marker ;
		y = end_marker;
	}
	else
	{
		x = start_marker;
		y = end_marker;	

	}




	
#if 0
	for(i = 0; i < length; i++)
	{
		if(max_value < abs(arr[i]))
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

	if(length < INTERVAL_SIZE)
	{
		printf("The number of sample in files are very less");
	}

	if(x < 0)
	{
		x = 1;
		y = INTERVAL_SIZE;
	}
	if(y >= length)
	{
		y = length - 2;
		x = y - INTERVAL_SIZE + 1;
	}
#endif

	for(i = x; i < y; i++)
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

	while(1)
	{
		if(feof(hamming_file))
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
		frame_array[i] = hamming_window[i] * sample_array[sliding_index+i];
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
	long long int i,j;
	
	for(i = 0 ; i < CODEBOOK_SIZE; i++)
	{
		for (j = 0 ; j < VECTOR_SIZE ;j++)
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

double tohkure_distance(long double training_vector[],long double codebook_vector[])
{
	double tohkura_weights[13] = {1.0, 3.0, 7.0, 13.0, 19.0, 22.0, 25.0, 33.0, 42.0, 50.0, 56.0, 61.0};
	long long int i = 0;
	double reference_value = 0.0;
	double test_value = 0.0;
	double distance = 0;

		for(i = 0; i <  VECTOR_SIZE; i++)
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
	
	for(i = 0 ; i < NO_OF_OBSERVATION; i++)
	{
		for(j = 0; j < VECTOR_SIZE; j++)
		{
			fscanf(input_fp_cepstral, "%Lf", &array_input_cepstral[i][j]);
		} 
	}
	
	for(i = 0; i < NO_OF_OBSERVATION; i++)
	{
		distance = INT_MAX;
		for(j = 0; j < CODEBOOK_SIZE; j++)
		{
			tohkure = tohkure_distance(array_input_cepstral[i], codebook[j]);
			if(distance > tohkure)
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
	for(i = 0; i < CODEBOOK_SIZE; i++)
	{
		for(j = 0 ; j < VECTOR_SIZE ; j++)
		{
			fscanf(codebook_fp ,"%Lf", &codebook[i][j]);
			//printf("%lf\t", codebook[i][j]);
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
	printf("\tCode to find the solution of problem 3 of HMM and train the model\n");
	printf("------------x--------------x---------------x------------\n");
	printf("Version is %ld.%ld\n\n", MAJOR_NUMBER, MINOR_NUMBER);
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
		
	/* Read Aij */
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		for(j = 1 ; j <= NO_OF_STATE; j++)
		{
			fscanf(input_fp_aij, "%Le", &array_aij[i][j]);
		}
	}
	
	/* Read Bjk */
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		for(j = 1 ; j <= NO_OF_OBSERVATION_SYMBOL; j++)
		{
			fscanf(input_fp_bjk, "%Le", &array_bjk[i][j]);
		}
	}

	/* Read PIi*/
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		fscanf(input_fp_pii, "%Le", &array_pii[i]);
	}


	/* Close file */
	fclose(input_fp_aij);
	fclose(input_fp_bjk);
	fclose(input_fp_pii);

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
	for( i = 1; i <= NO_OF_STATE; i++)
	{
		o1 = array_observation_sequence[i];
		array_alpha_ti[i][1] = array_pii[i] * array_bjk[i][o1] ;
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

	for(t = 1; t <= NO_OF_OBSERVATION - 1; t++)
	{
		for(j = 1; j <= NO_OF_STATE; j++)
		{
			temp_value = 0;
			for(i = 1; i <= NO_OF_STATE; i++)
			{
				temp_value += array_alpha_ti[i][t] * array_aij[i][j];
			}
			ot = array_observation_sequence[t + 1];
			array_alpha_ti[j][t + 1] =  temp_value * array_bjk[j][ot];
		}
	}

	for( i = 1; i <= NO_OF_OBSERVATION; i++)
	{
		for(j = 1; j <= NO_OF_STATE ; j++)
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
	for(i = 1 ;i <= NO_OF_STATE ; i++)
	{
		probability_of_O_given_model += array_alpha_ti[i][NO_OF_OBSERVATION];
	}

}

/*****************************************************
*  Name			: backward_step1_initialization		 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Initialize beta array_aij	matrix	 *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void backward_step1_initialization()
{
	long long int i,j;
	for(i = 1; i <= NO_OF_STATE; i++)
	{
		array_beta_ti[i][NO_OF_OBSERVATION] = 1.0;
	}
}

/*****************************************************
*  Name			: backward_step2_induction			 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Update beta array_aij	matrix		 *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void backward_step2_induction()
{
	long long int i, j, t, ot;
	output_fp_beta = fopen(HMM_BETA_MATRIX, "w+");

	for(t = NO_OF_OBSERVATION - 1; t >= 1; t--)
	{
		for(i = 1; i <= NO_OF_STATE; i++)
		{
			array_beta_ti[i][t]  = 0;
			for(j = 1; j <= NO_OF_STATE; j++)
			{
				ot = array_observation_sequence[t + 1];
				array_beta_ti[i][t] += array_aij[i][j] * array_bjk[j][ot] * array_beta_ti[j][t + 1];
			}
		}	
	}

	for( i = 1; i <= NO_OF_OBSERVATION; i++)
	{
		for(j = 1; j <= NO_OF_STATE ; j++)
		{
			fprintf(output_fp_beta, "%0.30Le\t", array_beta_ti[j][i]);	
		}
		fprintf(output_fp_beta, "\n");
	}
	fclose(output_fp_beta);
}

#ifndef OLD_GAMMA_SOLUTION2

/*****************************************************
*  Name			: calculate_gamma					 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Find the Gamma matrix from Xi upto *
*				  T - 1								 *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void calculate_gamma()
{

	long long int i, j, t;

	for(t = 1; t <= NO_OF_OBSERVATION - 1 ; t++)
	{
		for(i = 1; i <= NO_OF_STATE; i++)
		{
			array_gamma_ti[i][t] = 0;
			for(j = 1; j <= NO_OF_STATE; j++)
			{
				array_gamma_ti[i][t] += array_XI_ijt[i][j][t];
			}
		}
	}
	
	/* hard coded for last entry, not a good idea, that why another function is used */ 
	array_gamma_ti[1][85] = 0.00000000723699;
	array_gamma_ti[2][85] = 0.000000151977;
	array_gamma_ti[3][85] = 0.0000015767;
	array_gamma_ti[4][85] = 0.0000107745;
	array_gamma_ti[5][85] = 0.999987;

	/* print Gamma */
/*	printf("\nGAMMA matrix is \n");
	for(t = 1; t <= NO_OF_OBSERVATION ; t++)
	{
		for(i = 1; i <= NO_OF_STATE; i++)
		{
			printf("%Lf\t",array_gamma_ti[i][t]);
		}
		printf("\n");
	}
*/
}
#endif

#ifdef OLD_GAMMA_SOLUTION2
/*****************************************************
*  Name			: calculate_gamma					 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Find the Gamma matrix from alpha   *
*				  and Beta.							 *				
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void calculate_gamma()
{
	long double summation[NO_OF_OBSERVATION + 1] = {0.0};
	long int i, j, t;
	long double max_value = 0.0;

	for(t = 1; t <= NO_OF_OBSERVATION ; t++)
	{
		for(i = 1 ;	i <= NO_OF_STATE; i++)
		{
			summation[t] = summation[t] + array_alpha_ti[i][t] * array_beta_ti[i][t]; 
		}
	}

	for(t = 1; t <= NO_OF_OBSERVATION ; t++)
	{
		for(i = 1 ;	i <= NO_OF_STATE; i++)
		{
			array_gamma_ti[i][t] = (array_alpha_ti[i][t] * array_beta_ti[i][t]) / summation[t] ; 
		}
	}
}
#endif

/*****************************************************
*  Name			: viterbi_step1_initialization		 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Initialize psi and delta matrices  *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void viterbi_step1_initialization()
{
	long long int i, j, t;
	long long int ot;
	for(i = 1; i <= NO_OF_STATE ; i++)
	{
			ot = array_observation_sequence[1];
			array_delta_ti[i][1] = array_pii[i] * array_bjk[i][ot]; 
			array_psi_ti[i][1] = 0;
	}
}

/*****************************************************
*  Name			: viterbi_step2_recursion			 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Recursively find psi and delta     *
*				  matrices							 *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void viterbi_step2_recursion()
{
	long long int i, j, t;
	long long int ot;
	long double max_value = 0.0;

	for(j = 1; j <= NO_OF_STATE ; j++)
	{
		for(t = 2; t <= NO_OF_OBSERVATION; t++)
		{
			max_value = 0.0;
			for(i = 1; i <= NO_OF_STATE; i++)
			{
				if(max_value < array_delta_ti[i][t - 1] * array_aij[i][j])
				{
					max_value = array_delta_ti[i][t - 1] * array_aij[i][j];
					array_psi_ti[j][t] = i;
				}
			}
			ot = array_observation_sequence[t];
			array_delta_ti[j][t] = max_value * array_bjk[j][ot]; 
		}
	}
}

/*****************************************************
*  Name			: viterbi_step3_termination			 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Find Pstar, how well the model is. *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void viterbi_step3_termination()
{
	long double max_value = 0.0;
	long long int i;

	for(i = 1; i <= NO_OF_STATE; i++)
	{
		if(max_value < array_delta_ti[i][NO_OF_OBSERVATION])
		{
			max_value = Pstar = array_delta_ti[i][NO_OF_OBSERVATION];
			QstarT = i;
		}
	}

	if(check == 0)
	{
		last_Pstar = Pstar;
		check++;
	}
	else
	{
		//printf("The differece is %Le\n", (Pstar - last_Pstar));
		last_Pstar = Pstar;
		if((Pstar - last_Pstar) > THRESHOLD_PSTAR)
		{
			flag = 0;
		}
	}
}

/*****************************************************
*  Name			: viterbi_step4_path				 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Find the optimal state sequence	 *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void viterbi_step4_path()
{
	long long int t, i;
	long long int qstartplus1 = 0;

	array_state_sequence[NO_OF_OBSERVATION] = QstarT; 
	for(t = NO_OF_OBSERVATION - 1; t >= 1 ; t--)
	{
		qstartplus1 = array_state_sequence[t + 1]; 
		array_state_sequence[t] = array_psi_ti[qstartplus1][t + 1]; 
	}
	/* Dump the sequence in the file  */
	for(i = 1; i <= NO_OF_OBSERVATION; i++)
	{
		fprintf(output_state_sequence, "%ld ", array_state_sequence[i]);
	}
}

/*****************************************************
*  Name			: print_State_Sequence_Pstar		 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Print state sequence from viterbi. *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void print_State_Sequence_Pstar()
{
	long long int i;
	for(i = 1; i <= NO_OF_OBSERVATION; i++)
	{
		printf("%ld\t", array_state_sequence[i]);
	}
	printf("\nState Sequence\nP* value is: %0.30Le\n", Pstar);
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
	obs_seq_open = fopen(HMM_OBSERVATION_SEQUENCE, "a+");
	for(i = 1; i <= NO_OF_OBSERVATION; i++)
	{
		//printf("\t%ld", array_observation_sequence[i]);
		fprintf(obs_seq_open, "%ld\t", array_observation_sequence[i]);
	}
	//printf("\n");
	
	fprintf(obs_seq_open,"\n");
	fclose(obs_seq_open);
}

/*****************************************************
*  Name			: calculate_Xi						 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Calculate 3D Xi matrix as Sol3 part*
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void calculate_Xi()
{

	int i, j, t, ot;
	int counter = 0;

	for(t = 1; t <= NO_OF_OBSERVATION - 1 ; t++)
	{
		array_summation[t] = 0;
		for(i = 1; i <= NO_OF_STATE; i++)
		{
			for(j = 1; j <= NO_OF_STATE; j++)
			{
				ot = array_observation_sequence[t + 1];
				array_summation[t] += array_alpha_ti[i][t] * array_aij[i][j] * array_bjk[j][ot] * array_beta_ti[j][t + 1];
			}
		}
	}

	for(t = 1; t <= NO_OF_OBSERVATION - 1 ; t++)
	{
		for(i = 1; i <= NO_OF_STATE; i++)
		{
			for(j = 1; j <= NO_OF_STATE; j++)
			{
				ot = array_observation_sequence[t + 1];
				array_XI_ijt[i][j][t] = array_alpha_ti[i][t] * array_aij[i][j]  * array_bjk[j][ot] * array_beta_ti[j][t + 1] / array_summation[t] ;
				
			}
		}
	}
}

/*****************************************************
*  Name			: calculate_PII						 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Update PIi array as Sol3 part		 *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void calculate_PII()
{
	long long int i, j;
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		array_pii[i] = array_gamma_ti[i][1];
	}
}

/*****************************************************
*  Name			: calculate_AIJ						 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Update Aij array as Sol3 part		 *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void calculate_AIJ()
{
	long long int i, j, t;
	long double XIij = 0.0;
	long double GAMMAit = 0.0;

	for(i = 1; i <= NO_OF_STATE; i++)
	{
		for(j = 1; j <= NO_OF_STATE; j++)
		{
			XIij = 0.0;
			GAMMAit = 0.0;
			for(t = 1; t <= NO_OF_OBSERVATION - 1; t++)
			{
				XIij += array_XI_ijt[i][j][t];
				GAMMAit +=  array_gamma_ti[i][t];
			}
			array_aij[i][j] = XIij / GAMMAit;
		}
	}
}

/*****************************************************
*  Name			: calculate_BJK						 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Update Bjk array as Sol3 part		 *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void calculate_BJK()
{
	long long int t, j, k = 1;
	long long int index_j = 0, index_k = 0;
	long double numerator_gammajt = 0.0;
	long double denominator_gammajt = 0.0;
	long double b_summation = 0;

	for(j = 1; j <= NO_OF_STATE; j++)
	{
		for(k = 1; k <= NO_OF_OBSERVATION_SYMBOL; k++)
		{
			numerator_gammajt = 0.0;
			denominator_gammajt = 0.0;
			for(t = 1; t <= NO_OF_OBSERVATION; t++)
			{
				if(array_observation_sequence[t] == k)
				{
					numerator_gammajt += array_gamma_ti[j][t];
				}
			}
			for(t = 1; t <= NO_OF_OBSERVATION; t++)
			{
				denominator_gammajt +=  array_gamma_ti[j][t];
			}
			array_bjk[j][k] = numerator_gammajt / denominator_gammajt;
		}
	}
	threshold_to_Bjk();
}

/*****************************************************
*  Name			: threshold_to_Bjk					 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Apply minimum threshold to Bjk mat *
*				  so the value should not become 0	 *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void threshold_to_Bjk()
{
	long long int t, j, k = 1;
	long long int index_j = 0, index_k = 0;
	long double max_value = 0;
	long double difference = 0;

	for(j = 1; j <= NO_OF_STATE; j++)
	{
		max_value = 0;
		difference = 0;
		b_summation[j] = 0;
		for(k = 1; k <= NO_OF_OBSERVATION_SYMBOL; k++)
		{
			b_summation[j] += array_bjk[j][k];
			if(max_value <  array_bjk[j][k])
			{
				max_value = array_bjk[j][k];
				index_j = j;
				index_k = k;
			}
			if(array_bjk[j][k] < BJK_THRESHOLD)
			{
				difference += BJK_THRESHOLD - array_bjk[j][k];
				array_bjk[j][k] = BJK_THRESHOLD;
			}
		}
		array_bjk[index_j][index_k] = array_bjk[index_j][index_k] - difference;

	}
}

/*****************************************************
*  Name			: initialize_all_matrices			 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Initialize all matrix to value 0   *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void initialize_all_matrices()
{
	long long int i, j, k;
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		for(j = 1; j <= NO_OF_OBSERVATION; j++)
		{
			array_alpha_ti[i][j]= 0;
			array_beta_ti[i][j]	= 0;
			array_delta_ti[i][j]= 0;
			array_gamma_ti[i][j]= 0;
			array_psi_ti[i][j]	= 0;
		}
	}

	for(i = 1; i <= NO_OF_OBSERVATION; i++)
	{
		array_summation[i] = 0;
	}

	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		for(j = 1; j <= NO_OF_STATE; j++)
		{
			for(k = 1; k <= NO_OF_OBSERVATION; k++)
			{
				array_XI_ijt[i][j][k] = 0;
			}
		}
	}
	

}

/*****************************************************
*  Name			: update_model_to_file				 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: update the final model to file	 *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void update_model_to_file()
{
	long long int i, j;
	

	/* Update Aij */
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		for(j = 1 ; j <= NO_OF_STATE; j++)
		{
			fprintf(output_fp_aij, "%Le\t", array_aij[i][j]);
			//fprintf(output_fp_aij, "%Le\t", array_aij_copy[i][j]);
		}
		fprintf(output_fp_aij, "\n");
	}

	/* Update Bjk */	
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		for(j = 1 ; j <= NO_OF_OBSERVATION_SYMBOL; j++)
		{
			fprintf(output_fp_bjk, "%Le\t", array_bjk[i][j]);
			//fprintf(output_fp_bjk, "%Le\t", array_bjk_copy[i][j]);
		}
		fprintf(output_fp_bjk, "\n");
	}

	/* Update Pii */
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		fprintf(output_fp_pii, "%Lf\t", array_pii[i]);
	}


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
void debug_print()
{
	long int i, j, t;

#ifdef DEBUG_ALPHA
	printf("Alpha Matrix is:\n");
	for(i = 1 ; i <= NO_OF_OBSERVATION; i++)
	{
		for(j = 1 ; j <=  NO_OF_STATE; j++)
		{
			printf("%Le\t", array_alpha_ti[j][i]);
		}
		printf("\n");
	}
	printf("\n\n");
#endif

#ifdef DEBUG_BETA
	printf("Beta Matrix is:\n");
	for(i = 1 ; i <= NO_OF_OBSERVATION; i++)
	{
		for(j = 1 ; j <= NO_OF_STATE; j++)
		{
			printf("%Le\t", array_beta_ti[j][i]);
		}
		printf("\n");
	}
	printf("\n\n");
#endif

#ifdef DEBUG_AIJ
	printf("Aij Matrix is:\n");
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		for(j = 1 ; j <= NO_OF_STATE; j++)
		{
			printf("%Le\t", array_aij[i][j]);
		}
		printf("\n");
	}
	printf("\n\n");
#endif

#ifdef DEBUG_BJK
	printf("Bjk Matrix is:\n");
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		for(j = 1 ; j <= NO_OF_OBSERVATION_SYMBOL; j++)
		{
			printf("%Le\t", array_bjk[i][j]);
		}
		printf("\n");
	}
	printf("\n\n");
#endif

#ifdef DEBUG_PII
	printf("PIi Array is:\n");
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		printf("%0.30Lf\t", array_pii[i]);
	}
	printf("\n\n");
#endif

#ifdef DEBUG_DELTA
	printf("The Delta Matrix is: \n");
	for(i = 1; i <= NO_OF_STATE  ; i++)
	{
		for(j = 1 ;	j <= NO_OF_OBSERVATION; j++)
		{ 
			printf("%0.30Lf\n", array_delta_ti[i][j]);
		}
		printf("\n\n\n");
	}
#endif

#ifdef DEBUG_PSI
	printf("The psi Matrix is: \n");
	for(i = 1; i <= NO_OF_STATE  ; i++)
	{
		for(j = 1 ;	j <= NO_OF_OBSERVATION; j++)
		{ 
			printf("%ld ", array_psi_ti[i][j]);
		}
		printf("\n\n\n");
	}
#endif

#ifdef DEBUG_XII
	printf("The Xii 3D Matrix is:\n");
	for(t = 1; t <= NO_OF_OBSERVATION - 1 ; t++)
	{
		for(i = 1; i <= NO_OF_STATE; i++)
		{
			for(j = 1; j <= NO_OF_STATE; j++)
			{
				printf("%Le\t", array_XI_ijt[i][j][t]);
			}
			printf("\n");
		}
		printf("\n\n\n");
	}
#endif

#ifdef DEBUG_GAMMA
	for(t = 1; t <= NO_OF_OBSERVATION ; t++)
	{
		for(i = 1 ;	i <= NO_OF_STATE; i++)
		{
			printf("%Le\t", array_gamma_ti[i][t]); 
		}
		printf("\n");
	}

#endif
#ifdef B_SUMMATION_CHECK
	b_summation_check();
#endif

}

/*****************************************************
*  Name			: b_summation_check					 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Check validity of Bjk matrix.      *
*													 *
*  Return		: Nothing							 *
*													 *
******************************************************/
void b_summation_check()
{
	long long int i;
	for(i = 1; i <= NO_OF_STATE; i++)
	{
#ifdef B_SUMMATION_CHECK
		printf("The summation of B[Row %lld] is %0.60Le\n", i, b_summation[i]);
#endif		
if(b_summation[i] < 0.999998)
		{
			printf("The summation of B[Row %lld] is %0.60 This is not valid \n", i, b_summation[i]);
			printf("The code runs into some error, please restart it with different recording\n");
			Sleep(10000);
			exit(1);
		}
	}
}


/*****************************************************
*  Name			: read_Alpha_Beta					 *
*													 *
*  Arguments	: None								 *
*													 *
*  Description	: Read Alpha and Beta from file, if  *
*				  present							 *
*													 *
*  Return		: Nothing							 *
*													 *
*****************************************************/
void read_Alpha_Beta()
{
	long long int i, j;
	input_fp_alpha = fopen(HMM_ALPHA_MATRIX, "r+");
	input_fp_beta = fopen(HMM_BETA_MATRIX, "r+");

	/* Read Alpha Matrix */
	for(i = 1 ; i <= NO_OF_OBSERVATION; i++)
	{
		for(j = 1 ; j <= NO_OF_STATE; j++)
		{
			fscanf(input_fp_alpha, "%Lf", &array_alpha_ti[j][i]);
		}
	}

	/* Read Beta Matrix */
	for(i = 1 ; i <= NO_OF_OBSERVATION; i++)
	{
		for(j = 1 ; j <= NO_OF_STATE; j++)
		{
			fscanf(input_fp_beta, "%Lf", &array_beta_ti[j][i]);
		}
	}

	fclose(input_fp_beta);
	fclose(input_fp_alpha);
}

void add_model_arrays()
{
	long long int i, j;

	/* Update Aij copy */
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		for(j = 1 ; j <= NO_OF_STATE; j++)
		{
			 array_aij_copy[i][j] += array_aij[i][j];
	/*		 if(i == 1 && j == 1)
			 {
				printf("a[0][0] : %Lf\n ", array_aij[i][j]);
				printf("The sum is %Le\n",array_aij_copy[i][j]);
			 }
	*/
		}	
	}

	/* Update Bjk copy */	
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		for(j = 1 ; j <= NO_OF_OBSERVATION_SYMBOL; j++)
		{
			array_bjk_copy[i][j] += array_bjk[i][j];
		}
	}

	/* Update Pii copy */
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		array_pii_copy[i] += array_pii[i];
	}
}

void average_the_model()
{
	long long int i, j;

	/* Divide Aij copy */
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		for(j = 1 ; j <= NO_OF_STATE; j++)
		{
			 array_aij_copy[i][j] /= DIGIT_UTTERANCE_COUNT;
		}	
	}

	/* Divide Bjk copy */	
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		for(j = 1 ; j <= NO_OF_OBSERVATION_SYMBOL; j++)
		{
			array_bjk_copy[i][j] /= DIGIT_UTTERANCE_COUNT;
		}
	}

	/* Divide Pii copy */
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		array_pii_copy[i] /= DIGIT_UTTERANCE_COUNT;
	}
}


void update_average_model()
{

	long long int i, j;
	
	/* Update Aij */
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		for(j = 1 ; j <= NO_OF_STATE; j++)
		{
			fprintf(output_fp_average_aij, "%Le\t", array_aij_copy[i][j]);
			array_aij_copy[i][j] = 0;
		}
		fprintf(output_fp_average_aij, "\n");
	}

	/* Update Bjk */	
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		for(j = 1 ; j <= NO_OF_OBSERVATION_SYMBOL; j++)
		{
			fprintf(output_fp_average_bjk, "%Le\t", array_bjk_copy[i][j]);
			array_bjk_copy[i][j] = 0;
		}
		fprintf(output_fp_average_bjk, "\n");
	}

	/* Update Pii */
	for(i = 1 ; i <= NO_OF_STATE; i++)
	{
		fprintf(output_fp_average_pii, "%Lf\t", array_pii_copy[i]);
		array_pii_copy[i] = 0;
	}

}

void read_Observation_Sequence()
{
	long long int i;
	
	/* Read Observation Sequence */
	for(i = 1 ; i <= NO_OF_OBSERVATION; i++)
	{
		fscanf(input_fp_observation_sequence, "%ld", &array_observation_sequence[i]);
	}
}

/*****************************************************
*  Name			: _tmain							 *
*													 *
*  Arguments	: No of argument, argument values	 *
*													 *
*  Description	: Driving function to find Sol2 HMM  *
*													 *
*  Return		: 0									 *
*													 *
*****************************************************/
int _tmain(int argc, _TCHAR* argv[])
{
	char wait;
	long int i, j;
	long int iteration_count = 0;
	char intermediate_file_index[3];
	int digit;
	int digit_utterance;
	int iteration_count_for_update;
	

	long long int no_of_samples = 0;
	double max_value = 0.0;
	double ratio = 0.0;
	long long int ret = 0;
	double DC_SHIFT_VALUE = 0.0;
	double normalized_value = NORMALIZE_VALUE;
	char enter;
	double arr[100000];
	double sample_array[15000] = {0};
	double hamming_window[321] = {0};
	double frame_array[321] = {0};
	long long int start = 0, end = 0;
	long double arr_Ri[30] = {0};
	long double arr_Ai[30] = {0};
	long double arr_Ci[30] = {0};
	long double arr_Cdash[30] = {0};
	long long int sliding_index = 0;
	long long int length = 0;
	long long int no_of_frames = NO_OF_OBSERVATION;
	long double find_vowel_distance[6] = {0.0};
	char reference_file_name[300];
	long long int minimum_index = 0;
	long double minimum = 999999.0;
	long double max_probability = 0;
	long int digit_spoken = 0;
	char recording[1000];
	char temp_digit[1000];
	int correctness= 0;
	int user_choice;
	char color_spoken[100];
	long int color_id = 0;
	char input_test_name[1000];
	FILE *obs_seq_open ;
	read_codebook();

	for( int temp = 0 ; temp < 8 ; temp++)
	{
		strcpy(HMM_OBSERVATION_SEQUENCE, "Input/HMM_OBSERVATION_SEQUENCE_");
		sprintf(intermediate_file_index, "%ld", temp);
		strcat(HMM_OBSERVATION_SEQUENCE, intermediate_file_index);
		strcat(HMM_OBSERVATION_SEQUENCE, ".txt");
		
		printf("The name of file is %s", HMM_OBSERVATION_SEQUENCE);
		printf("\n\n\t Maintain silence for three seconds  \n");
		system("Recording_Module.exe 3 silence.wav silence_file.txt");
		for( j = 0 ; j < 15 ; j++)
		{
			printf("\nSpeak %s color\n", sample_color[temp]);


			/* Taking input file name */
			strcpy(recording, "184101035_");
			sprintf(temp_digit, "%d", temp);
			strcat(recording, temp_digit);
			strcat(recording, "_");
			sprintf(temp_digit, "%02d", j+1);
			strcat(recording, temp_digit);
			strcat(recording, ".txt");
			strcpy(input_test_name, recording);
			strcpy(temp_digit, recording);
			strcpy(recording, "Recording_Module.exe 3 input_file.wav ");
			strcat(recording, temp_digit);
			printf("%s", recording);
			system(recording);

			printf("Input test name is %s\n", input_test_name);
		//	strcpy(input_test_name, "Input/");
		//	strcat(input_test_name, intermediate);
			silence_fp =	fopen("silence_file.txt","r+");
			//input_fp =		fopen(input_test_name, "r+");
			input_fp =		fopen(input_test_name, "r+");
			operation_fp =	fopen("sample_output.txt", "w+");
			dump_Ri =		fopen("dump_Ri_test.txt", "w+");
			dump_Ai =		fopen("dump_Ai_test.txt", "w+");
			dump_Ci =		fopen("dump_Ci_test.txt", "w+");
			dump_Cdash   =	fopen("dump_Cdash_test.txt", "w+");
			sample_file_fp =fopen("Sample_file.txt", "w+");
			hamming_file =	fopen(HAMMING_WINDOW, "r+");
	

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
			operation_fp =	fopen("sample_output.txt", "r+");
			copy_to_array(operation_fp, sample_array, &length);
			

			for(i = 0; i < FRAME_SIZE; i++)
			{
				frame_array[i] = sample_array[i];
			}

			for(i = 0 ; i < no_of_frames ; i++)
			{
				apply_hamming_window(hamming_window, frame_array, sample_array, sliding_index + i*80);
				do_autocorelation(frame_array, arr_Ri, length, dump_Ri);
				do_durbin(arr_Ri, dump_Ai, arr_Ai);
				do_capstral(arr_Ai, arr_Ri, arr_Ci, dump_Ci);
				apply_sine_window(arr_Cdash, arr_Ci, dump_Cdash);
			}

			fclose(dump_Cdash);
			dump_Cdash   =	fopen("dump_Cdash_test.txt", "r+");

			find_observation_sequence();
			print_Observation_Sequence();
			fclose(dump_Cdash);
			fclose(silence_fp);
			fclose(input_fp);		
			fclose(operation_fp);
			fclose(dump_Ri);		
			fclose(dump_Ai);		
			fclose(dump_Ci);		
			//fclose(sample_file_fp);
			fclose(hamming_file);
		}
		//fclose(obs_seq_open);
	}




	/* Open file */
	output_state_sequence = fopen(HMM_STATE_SEQUENCE, "w+");
	print_info();
	printf("Training started\n");
	for (iteration_count_for_update = 0; iteration_count_for_update < NO_OF_TIMES_TO_AVERAGE; iteration_count_for_update++)
	{
		printf("\n--------- Iteration No %d -----------------\n", iteration_count_for_update + 1);
		for(digit = 0; digit <= DIGIT; digit++)
		{
			printf("For digit %d\n", digit);
			strcpy(HMM_OBSERVATION_SEQUENCE, "Input/HMM_OBSERVATION_SEQUENCE_");
			sprintf(intermediate_file_index, "%ld", digit);
			strcat(HMM_OBSERVATION_SEQUENCE, intermediate_file_index);
			strcat(HMM_OBSERVATION_SEQUENCE, ".txt");
			
			printf("The file name is %s\n", HMM_OBSERVATION_SEQUENCE);
		
			input_fp_observation_sequence = fopen(HMM_OBSERVATION_SEQUENCE, "r+");
			//read_Alpha_Beta();
			
			if(iteration_count_for_update == 0)
			{
				strcpy(HMM_AIJ, "HMM_AIJ.txt");
				strcpy(HMM_BJK, "HMM_BJK.txt");
				strcpy(HMM_PII, "HMM_PII.txt");	
			}
			else
			{
				strcpy(HMM_AIJ, "Output/Average_Models/Average_A_I_");
				sprintf(intermediate_file_index, "%ld", iteration_count_for_update);
				strcat(HMM_AIJ, intermediate_file_index);
				strcat(HMM_AIJ, "_D_");
				sprintf(intermediate_file_index, "%ld", digit);
				strcat(HMM_AIJ, intermediate_file_index);
				strcat(HMM_AIJ, ".txt");

				strcpy(HMM_BJK, "Output/Average_Models/Average_B_I_");
				sprintf(intermediate_file_index, "%ld", iteration_count_for_update);
				strcat(HMM_BJK, intermediate_file_index);
				strcat(HMM_BJK, "_D_");
				sprintf(intermediate_file_index, "%ld", digit);
				strcat(HMM_BJK, intermediate_file_index);
				strcat(HMM_BJK, ".txt");

				strcpy(HMM_PII, "Output/Average_Models/Average_PI_I_");
				sprintf(intermediate_file_index, "%ld", iteration_count_for_update);
				strcat(HMM_PII, intermediate_file_index);
				strcat(HMM_PII, "_D_");
				sprintf(intermediate_file_index, "%ld", digit);
				strcat(HMM_PII, intermediate_file_index);
				strcat(HMM_PII, ".txt");

			}
			//printf("The file for initial model are \nA: %s\nB: %s\nP: %s\n", HMM_AIJ, HMM_BJK,HMM_PII);
			
			for(digit_utterance = 1 ; digit_utterance <= DIGIT_UTTERANCE_COUNT; digit_utterance++)
			{
				iteration_count = 0;
				read_Model();
				/* Read Observation sequence */
				read_Observation_Sequence();

				strcpy(HMM_AIJ_FINAL, "Output/Model_of_");
				sprintf(intermediate_file_index, "%ld", digit);
				strcat(HMM_AIJ_FINAL, intermediate_file_index);
				strcat(HMM_AIJ_FINAL, "/A");
				sprintf(intermediate_file_index, "%ld", digit_utterance);
				strcat(HMM_AIJ_FINAL, intermediate_file_index);
				strcat(HMM_AIJ_FINAL, ".txt");

				strcpy(HMM_BJK_FINAL, "Output/Model_of_");
				sprintf(intermediate_file_index, "%ld", digit);
				strcat(HMM_BJK_FINAL, intermediate_file_index);
				strcat(HMM_BJK_FINAL, "/B");
				sprintf(intermediate_file_index, "%ld", digit_utterance);
				strcat(HMM_BJK_FINAL, intermediate_file_index);
				strcat(HMM_BJK_FINAL, ".txt");

				strcpy(HMM_PII_FINAL, "Output/Model_of_");
				sprintf(intermediate_file_index, "%ld", digit);
				strcat(HMM_PII_FINAL, intermediate_file_index);
				strcat(HMM_PII_FINAL, "/PI");
				sprintf(intermediate_file_index, "%ld", digit_utterance);
				strcat(HMM_PII_FINAL, intermediate_file_index);
				strcat(HMM_PII_FINAL, ".txt");

			
				/* Open Files to wrtie */
				output_fp_aij = fopen(HMM_AIJ_FINAL, "w+"); 
				output_fp_bjk = fopen(HMM_BJK_FINAL, "w+");
				output_fp_pii = fopen(HMM_PII_FINAL, "w+");

				while(iteration_count < NO_OF_ITERATION_TO_TRAIN)
				{

					/*  SOLUTION 1 */
					/* Forward Procedure */
					forward_step1_initialization();
					forward_step2_induction();
					forward_step3_termination();
				
					/* Backward Procedure */
					backward_step1_initialization();
					backward_step2_induction();

					/*  SOLUTION 2 */
					/* Viterbi implementation */
					viterbi_step1_initialization();
					viterbi_step2_recursion();
					viterbi_step3_termination();
					viterbi_step4_path();

					if(flag == 1)
					{	
						printf("\nThe break statement is reached\n");
						break;
					}
#ifdef DEBUG_OBSERVATION_SEQUENCE_PRINT
					printf("Observation Sequence\n");
					print_Observation_Sequence();
#endif
#ifdef DEBUG_STATE_SEQUENCE_PRINT
					print_State_Sequence_Pstar();
#endif
		
					/* SOLUTION 3 */
					/* Expectation Maximization Implementation */
					calculate_Xi();
					calculate_gamma();
					calculate_PII();
					calculate_AIJ();
					calculate_BJK();
					b_summation_check();
					iteration_count++;
					//initialize_all_matrices();
				}
			
				/* Update new model to file */
				add_model_arrays();
				update_model_to_file();

				fclose(output_fp_aij);
				fclose(output_fp_bjk);
				fclose(output_fp_pii);
			}
			average_the_model();

			strcpy(HMM_AVERAGE_AIJ, "Output/Average_Models/Average_A_I_");
			sprintf(intermediate_file_index, "%ld", iteration_count_for_update + 1);
			strcat(HMM_AVERAGE_AIJ, intermediate_file_index);
			strcat(HMM_AVERAGE_AIJ, "_D_");
			sprintf(intermediate_file_index, "%ld", digit);
			strcat(HMM_AVERAGE_AIJ, intermediate_file_index);
			strcat(HMM_AVERAGE_AIJ, ".txt");

			strcpy(HMM_AVERAGE_BJK, "Output/Average_Models/Average_B_I_");
			sprintf(intermediate_file_index, "%ld", iteration_count_for_update + 1);
			strcat(HMM_AVERAGE_BJK, intermediate_file_index);
			strcat(HMM_AVERAGE_BJK, "_D_");
			sprintf(intermediate_file_index, "%ld", digit);
			strcat(HMM_AVERAGE_BJK, intermediate_file_index);
			strcat(HMM_AVERAGE_BJK, ".txt");

			strcpy(HMM_AVERAGE_PII, "Output/Average_Models/Average_PI_I_");
			sprintf(intermediate_file_index, "%ld", iteration_count_for_update + 1);
			strcat(HMM_AVERAGE_PII, intermediate_file_index);
			strcat(HMM_AVERAGE_PII, "_D_");
			sprintf(intermediate_file_index, "%ld", digit);
			strcat(HMM_AVERAGE_PII, intermediate_file_index);
			strcat(HMM_AVERAGE_PII, ".txt");

			if((iteration_count_for_update + 1) == NO_OF_TIMES_TO_AVERAGE)
			{
				strcpy(HMM_AVERAGE_AIJ, "Output/Average_Models/a_");
				sprintf(intermediate_file_index, "%ld", digit);
				strcat(HMM_AVERAGE_AIJ, intermediate_file_index);
				strcat(HMM_AVERAGE_AIJ, ".txt");

				strcpy(HMM_AVERAGE_BJK, "Output/Average_Models/b_");
				sprintf(intermediate_file_index, "%ld", digit);
				strcat(HMM_AVERAGE_BJK, intermediate_file_index);
				strcat(HMM_AVERAGE_BJK, ".txt");

				strcpy(HMM_AVERAGE_PII, "Output/Average_Models/p_");
				sprintf(intermediate_file_index, "%ld", digit);
				strcat(HMM_AVERAGE_PII, intermediate_file_index);
				strcat(HMM_AVERAGE_PII, ".txt");
			}
			output_fp_average_aij = fopen(HMM_AVERAGE_AIJ, "w+");
			output_fp_average_bjk = fopen(HMM_AVERAGE_BJK, "w+");
			output_fp_average_pii = fopen(HMM_AVERAGE_PII, "w+");

			
			update_average_model();

			fclose(output_fp_average_aij);
			fclose(output_fp_average_bjk);
			fclose(output_fp_average_pii);
		}
	}

	fclose(input_fp_observation_sequence);
	printf("Training is completed\n");
	printf("The final model are present in Output\\Average_Models\ Folder");
	
#ifdef DEBUG_PRINT
	debug_print();
#endif

exit:
	printf("\nEnter any key to continue....\n");
	scanf("%c", &wait);
	return 0;
}




