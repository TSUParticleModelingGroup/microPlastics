//Optimized using shared memory and on chip memory 																																			
// nvcc microPlastics.cu -o microPlastics -lglut -lm -lGLU -lGL
//To stop hit "control c" in the window you launched it from.
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
using namespace std;

FILE* ffmpeg;

#define BOLD_ON  "\e[1m"
#define BOLD_OFF   "\e[m"

#define PI 3.141592654
#define BLOCK 256

FILE* MovieFile;
int* Buffer;
int MovieFlag; // 0 movie off, 1 movie on

// Globals to be read in from parameter file.
int NumberOfMicroPlastics;
double DensityOfMicroPlasticMin;
double DensityOfMicroPlasticMax;
float DiameterOfMicroPlasticMin;
float DiameterOfMicroPlasticMax;

int NumberOfPolymerChains;
int PolymersChainLengthMin;
int PolymersChainLengthMax;

float PolymersConnectionLength;
double DensityOfPolymer;
float DiameterOfPolymer;

float BeakerRadius; //4900.0;
float FluidHeight; //118000.0;

float FluidDensity;
float Drag;

float TotalRunTime;
float Dt;
int DrawRate;
int PrintRate;

float PolymerRed;
float PolymerGreen;
float PolymerBlue;

float MicroPlasticRed;
float MicroPlasticGreen;
float MicroPlasticBlue;

// Other Globals
int Pause;
int ViewFlag; // 0 orthoganal, 1 fulstum
int TopView;
int NumberOfBodies;
int NumberOfPolymers;
float4 *BodyPosition, *BodyVelocity, *BodyForce;
float4 *BodyPositionGPU, *BodyVelocityGPU, *BodyForceGPU;
int *PolymerChainLength;
int *PolymerConnectionA, *PolymerConnectionB;
int *PolymerConnectionAGPU, *PolymerConnectionBGPU;
curandState_t* DevStates;
dim3 Blocks, Grids;
int DrawTimer, PrintTimer;
float RunTime;
float4 CenterOfSimulation;
float4 DistanceFromCenter = make_float4(0.0, 0.0, 0.0, 0.0);
float4 AngleOfSimulation;
float ShakeItUpMag;

int DebugFlag;
int RadialConfinementViewingAids;
int StirFlag;
float StirAngularVelosity;
float Theta;
int ShakeItUpFlag;

// Window globals
static int Window;
int XWindowSize;
int YWindowSize;
double Near;
double Far;
double EyeX;
double EyeY;
double EyeZ;
double CenterX;
double CenterY;
double CenterZ;
double UpX;
double UpY;
double UpZ;

#include "./forceFunctions.h"

// Prototyping functions
void readSimulationParameters();
void setNumberOfBodies();
void allocateMemory();
void setInitailConditions();
void drawPicture();
void nBody();
void errorCheck(const char*);
void terminalPrint();
void setup();

__global__ void init_curand(unsigned int, curandState_t*);
__global__ void getForces(curandState_t*, float4 *, float4 *, float4 *, int *, int *, float, int, int, float, float, float, int, float, int, float );
__global__ void getForcesSetup(curandState_t* , float4 *, float4 *, float4 *, int *, int *, float, int, int, float, float, float, int, float, int );
__global__ void moveBodies(float4 *pos, float4 *, float4 *, float, float, int);

#include "./callBackFunctions.h"

void readSimulationParameters()
{
	ifstream data;
	string name;
	
	data.open("./simulationSetup");
	
	if(data.is_open() == 1)
	{
		getline(data,name,'=');
		data >> NumberOfMicroPlastics;
		
		getline(data,name,'=');
		data >> DensityOfMicroPlasticMin;

		getline(data,name,'=');
		data >> DensityOfMicroPlasticMax;
		
		getline(data,name,'=');
		data >> DiameterOfMicroPlasticMin;

		getline(data,name,'=');
		data >> DiameterOfMicroPlasticMax;
		
		getline(data,name,'=');
		data >> NumberOfPolymerChains;
		
		getline(data,name,'=');
		data >> PolymersChainLengthMin;

		getline(data,name,'=');
		data >> PolymersChainLengthMax;
		
		getline(data,name,'=');
		data >> PolymersConnectionLength;
		
		getline(data,name,'=');
		data >> DensityOfPolymer;
		
		getline(data,name,'=');
		data >> DiameterOfPolymer;
		
		getline(data,name,'=');
		data >> BeakerRadius;

		getline(data,name,'=');
		data >> FluidHeight;

		getline(data,name,'=');
		data >> FluidDensity;

		getline(data,name,'=');
		data >> Drag;
		
		getline(data,name,'=');
		data >> TotalRunTime;
		
		getline(data,name,'=');
		data >> Dt;
		
		getline(data,name,'=');
		data >> DrawRate;
		
		getline(data,name,'=');
		data >> PrintRate;

		getline(data,name,'=');
		data >> PolymerRed;

		getline(data,name,'=');
		data >> PolymerGreen;

		getline(data,name,'=');
		data >> PolymerBlue;

		getline(data,name,'=');
		data >> MicroPlasticRed;

		getline(data,name,'=');
		data >> MicroPlasticGreen;

		getline(data,name,'=');
		data >> MicroPlasticBlue;
		
	}
	else
	{
		printf("\nTSU Error could not open simulationSetup file\n");
		exit(0);
	}
	data.close();
	
	if(DebugFlag == 1)
	{
	//prinf all the parameters
		printf("\n\n Number of MicroPlastics = %d", NumberOfMicroPlastics);
		printf("\n DensityOfMicroPlasticMin = %f", DensityOfMicroPlasticMin);
		printf("\n DensityOfMicroPlasticMax = %f", DensityOfMicroPlasticMax);
		printf("\n DiameterOfMicroPlasticMin = %f", DiameterOfMicroPlasticMin);
		printf("\n DiameterOfMicroPlasticMax = %f", DiameterOfMicroPlasticMax);
		printf("\n NumberOfPolymerChains = %d", NumberOfPolymerChains);
		printf("\n PolymersChainLengthMin = %d", PolymersChainLengthMin);
		printf("\n PolymersChainLengthMax = %d", PolymersChainLengthMax);
		printf("\n PolymersConnectionLength = %f", PolymersConnectionLength);
		printf("\n DensityOfPolymer = %f", DensityOfPolymer);
		printf("\n DiameterOfPolymer = %f", DiameterOfPolymer);
		printf("\n BeakerRadius = %f", BeakerRadius);
		printf("\n FluidHeight = %f", FluidHeight);
		printf("\n FluidDensity = %f", FluidDensity);
		printf("\n Drag = %f", Drag);
		printf("\n TotalRunTime = %f", TotalRunTime);
		printf("\n Dt = %f", Dt);
		printf("\n DrawRate = %d", DrawRate);
		printf("\n PrintRate = %d", PrintRate);
		printf("\n PolymerRed = %f", PolymerRed);
		printf("\n PolymerGreen = %f", PolymerGreen);
		printf("\n PolymerBlue = %f", PolymerBlue);
		printf("\n MicroPlasticRed = %f", MicroPlasticRed);
		printf("\n MicroPlasticGreen = %f", MicroPlasticGreen);
		printf("\n MicroPlasticBlue = %f", MicroPlasticBlue);
	}
	printf("\n Parameter file has been read.\n");
}

void setNumberOfBodies()
{
	time_t t;
	
	PolymerChainLength = (int*)malloc(NumberOfPolymerChains*sizeof(int));
	
	srand((unsigned) time(&t));
	for(int i = 0; i < NumberOfPolymerChains; i++)
	{
		PolymerChainLength[i] = ((float)rand()/(float)RAND_MAX)*(PolymersChainLengthMax - PolymersChainLengthMin) + PolymersChainLengthMin;
		//printf("\n PolymerChainLength[%d] = %d", i, PolymerChainLength[i]);	
	}
	
	NumberOfPolymers = 0;
	for(int i = 0; i < NumberOfPolymerChains; i++)
	{
		NumberOfPolymers += PolymerChainLength[i];	
	}
	
	NumberOfBodies = NumberOfMicroPlastics + NumberOfPolymers;
	
	if(DebugFlag == 1)
	{
		printf("\n\n Number of Polymers = %d", NumberOfPolymers);
		printf("\n Number of MicroPlastics = %d", NumberOfMicroPlastics);
		printf("\n Total number of bodies = %d", NumberOfBodies);
	}
	
	printf("\n Number of bodies has been set.\n");
}

void allocateMemory()
{
	Blocks.x = BLOCK;
	Blocks.y = 1;
	Blocks.z = 1;
	
	Grids.x = (NumberOfBodies - 1)/Blocks.x + 1;
	Grids.y = 1;
	Grids.z = 1;
	
	BodyPosition = (float4*)malloc(NumberOfBodies*sizeof(float4));
	BodyVelocity = (float4*)malloc(NumberOfBodies*sizeof(float4));
	BodyForce    = (float4*)malloc(NumberOfBodies*sizeof(float4));
	
	PolymerConnectionA    = (int*)malloc(NumberOfPolymers*sizeof(int));
	PolymerConnectionB    = (int*)malloc(NumberOfPolymers*sizeof(int));
	
	cudaMalloc( (void**)&BodyPositionGPU, NumberOfBodies *sizeof(float4));
	errorCheck("cudaMalloc BodyPositionGPU");
	cudaMalloc( (void**)&BodyVelocityGPU, NumberOfBodies *sizeof(float4));
	errorCheck("cudaMalloc BodyDiameterOfBodyVelocityGPU");
	cudaMalloc( (void**)&BodyForceGPU, NumberOfBodies *sizeof(float4));
	errorCheck("cudaMalloc BodyForceGPU");
	
	cudaMalloc( (void**)&PolymerConnectionAGPU, NumberOfPolymers *sizeof(int));
	errorCheck("cudaMalloc BodyForceGPU");
	cudaMalloc( (void**)&PolymerConnectionBGPU, NumberOfPolymers *sizeof(int));
	errorCheck("cudaMalloc BodyForceGPU");
	
	cudaMalloc((void**)&DevStates, NumberOfBodies * sizeof(curandState_t));
	
	printf("\n Memory has been allocated.\n");
}

/******************************************************************************
 This function does most of the nbody work when shaking up the polymers for setup.
*******************************************************************************/
__global__ void getForcesSetup(curandState_t* states, float4 *pos, float4 *vel, float4 *force, int *linkA, int *linkB, float length, int nPolymer, int nPlastics, float beakerRadius, float fluidHeight, float fluidDensity, int stirFlag, float theta, int shakeItUpFlag)
{
	int myId, yourId;
	float4 forceVector, forceVectorSum;
	float4 posMe, posYou;
	
	myId = threadIdx.x + blockDim.x*blockIdx.x;
    	if(myId < nPolymer)
    	{
		posMe.x = pos[myId].x;
		posMe.y = pos[myId].y;
		posMe.z = pos[myId].z;
		posMe.w = pos[myId].w;
		
		forceVectorSum.x = 0.0;
		forceVectorSum.y = 0.0;
		forceVectorSum.z = 0.0;
		
		for(yourId = 0; yourId < nPolymer; yourId++)
		{	
			if(yourId != myId) // Making sure you are not working on youself.
			{
				posYou.x = pos[yourId].x;
				posYou.y = pos[yourId].y;
				posYou.z = pos[yourId].z;
				posYou.w = pos[yourId].w;
			
				if(myId < nPolymer && yourId < nPolymer)
				{
					// Polymer-polymer force
					forceVector = getPolymerPolymerForce(posMe, posYou, linkA[myId], linkB[myId], yourId, length, myId);
				}
				
			    	forceVectorSum.x += forceVector.x;
			    	forceVectorSum.y += forceVector.y;
			    	forceVectorSum.z += forceVector.z;
		    	}
		}
		
		// This adds on the forces to keep the bodies in the container.
		forceVector = getContainerForces(posMe, beakerRadius, fluidHeight);
		forceVectorSum.x += forceVector.x;
		forceVectorSum.y += forceVector.y;
		forceVectorSum.z += forceVector.z;
		
		// Tranfering all the forces to my force function
		force[myId].x += forceVectorSum.x;
		force[myId].y += forceVectorSum.y;
		force[myId].z += forceVectorSum.z;
    	}
}

void polymerShakeUp(float4 *pos, float4 *vel, float4 *force, int *linkA, int *linkB, float length, int n, float drag, float dt, float beakerRadius, float fluidHeight)
{
	float magx, magy, magz, mag;
	float dragTemp = drag;
	
	mag = 10.0;
	float stopTime = 2.0;
	float time = 0;
	DrawTimer = 0;
	
	cudaMemcpy( PolymerConnectionAGPU, PolymerConnectionA, NumberOfPolymers*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( PolymerConnectionBGPU, PolymerConnectionB, NumberOfPolymers*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( BodyPositionGPU, BodyPosition, NumberOfBodies*sizeof(float4), cudaMemcpyHostToDevice );
	cudaMemcpy( BodyVelocityGPU, BodyVelocity, NumberOfBodies*sizeof(float4), cudaMemcpyHostToDevice );
	cudaMemcpy( BodyForceGPU, BodyForce, NumberOfBodies*sizeof(float4), cudaMemcpyHostToDevice );
		
	while(time < stopTime)
	{
		for(int i = 0; i < n; i++)
		{
			force[i].x = 0.0;
			force[i].y = 0.0;
			force[i].z = 0.0;
		}
		
		for(int i = 0; i < n; i++)
		{
			mag = 10.0;
			if(time < 1.0)
			{
				dragTemp = 0.001;
				if(linkA[i] == -1) 
				{
					force[i].y -= 500.0;
					force[i].x -= 10.0;
				}
				if(linkB[i] == -1) 
				{
					force[i].y += 500.0;
					force[i].x += 10.0;
				}
				
				magx = mag*((float)rand()/RAND_MAX*2.0 - 1.0);
				magy = mag*((float)rand()/RAND_MAX*2.0 - 1.0);
				magz = mag*((float)rand()/RAND_MAX*2.0 - 1.0);
				
				force[i].x += magx;
				force[i].y += magy;
				force[i].z += magz;
			}
			else if(time < 1.5)
			{
				dragTemp = 0.01;
			}
			else
			{

				dragTemp = drag;

			}	

		}
		
		cudaMemcpy( BodyForceGPU, BodyForce, NumberOfBodies*sizeof(float4), cudaMemcpyHostToDevice );
		getForcesSetup<<<Grids, Blocks>>>(DevStates, BodyPositionGPU, BodyVelocityGPU, BodyForceGPU, PolymerConnectionAGPU, PolymerConnectionBGPU, PolymersConnectionLength, NumberOfPolymers, NumberOfMicroPlastics, BeakerRadius, FluidHeight, FluidDensity, StirFlag, Theta, ShakeItUpFlag);
		errorCheck("getForcesSetup");
		
		moveBodies<<<Grids, Blocks>>>(BodyPositionGPU, BodyVelocityGPU, BodyForceGPU, dragTemp, Dt, NumberOfBodies);
		errorCheck("moveBodies");
		cudaMemcpy( BodyForce, BodyForceGPU, NumberOfBodies*sizeof(float4), cudaMemcpyDeviceToHost );
			
		time += dt;
	}
	
	// Taking the shaken up polymers back to the GPU so that the microplastics can be added on.
	cudaMemcpy( BodyPosition, BodyPositionGPU, NumberOfBodies*sizeof(float4), cudaMemcpyDeviceToHost );
	cudaMemcpy( BodyVelocity, BodyVelocityGPU, NumberOfBodies*sizeof(float4), cudaMemcpyDeviceToHost );
	cudaMemcpy( BodyForce, BodyForceGPU, NumberOfBodies*sizeof(float4), cudaMemcpyDeviceToHost );
	
	printf("\n Polymers have been shoken up.\n");
}

void setInitailConditions()
{
	time_t t;
	srand((unsigned) time(&t));
	double density;
	double angle;
	double dx, dy, dz, d2, d;
	int k;
	int index;
	int test;
	double TotalPolymerLength;
	double spaceBetweenPolymerCenters;
	double startX,startY,startZ;
	
	// Zeroing out everything just for safety
	for(int i = 0; i < NumberOfBodies; i++)
	{
		BodyPosition[i].x = 0.0;
		BodyPosition[i].y = 0.0;
		BodyPosition[i].z = 0.0;
		BodyPosition[i].w = 0.0;
		
		BodyVelocity[i].x = 0.0;
		BodyVelocity[i].y = 0.0;
		BodyVelocity[i].z = 0.0;
		BodyVelocity[i].w = 0.0;
		
		BodyForce[i].x = 0.0;
		BodyForce[i].y = 0.0;
		BodyForce[i].z = 0.0;
		BodyForce[i].w = 0.0;
	}
	
	// Loading velocity, diameter, density, and mass of polymers
	for(int i = 0; i < NumberOfPolymers; i++)
	{
		BodyVelocity[i].x = 0.0;
		BodyVelocity[i].y = 0.0;
		BodyVelocity[i].z = 0.0;
		
		// Setting diameter
		BodyPosition[i].w = DiameterOfPolymer;	
		
		// Setting density
		BodyVelocity[i].w = DensityOfPolymer;
		
		// Setting mass
		BodyForce[i].w = DensityOfPolymer*(4.0/3.0)*PI*(BodyPosition[i].w/2.0)*(BodyPosition[i].w/2.0)*(BodyPosition[i].w/2.0);
	}
	
	// Setting velocity, diameter, density, and mass of microplastics
	for(int i = NumberOfPolymers; i < NumberOfBodies; i++)
	{
		BodyVelocity[i].x = 0.0;
		BodyVelocity[i].y = 0.0;
		BodyVelocity[i].z = 0.0;
		
		// Setting diameter
		BodyPosition[i].w = ((double)rand()/(double)RAND_MAX)*(DiameterOfMicroPlasticMax - DiameterOfMicroPlasticMin) + DiameterOfMicroPlasticMin;
		
		// Setting density
		density = ((double)rand()/(double)RAND_MAX)*(DensityOfMicroPlasticMax - DensityOfMicroPlasticMin) + DensityOfMicroPlasticMin;
		BodyVelocity[i].w = density;
		
		// Setting mass
		BodyForce[i].w = density*(4.0/3.0)*PI*(BodyPosition[i].w/2.0)*(BodyPosition[i].w/2.0)*(BodyPosition[i].w/2.0);	
	}
	
	// Setting intial positions of polymers
	spaceBetweenPolymerCenters = PolymersConnectionLength+DiameterOfPolymer;
	k = 0;
	for(int i = 0; i < NumberOfPolymerChains; i++)
	{
		test = 0;
		while(test == 0)
		{
			angle = 2.0*PI*(double)rand()/(double)RAND_MAX;
			BodyPosition[k].x = ((double)rand()/(double)RAND_MAX)*BeakerRadius * cos(angle);
			BodyPosition[k].z = ((double)rand()/(double)RAND_MAX)*BeakerRadius * sin(angle);
			
			test = 1;
			index = 0;
			for(int j = 0; j < i; j++)
			{
				// Checking against the leading element of the polymer chain.
				dx = BodyPosition[k].x - BodyPosition[index].x;
				dz = BodyPosition[k].z - BodyPosition[index].z;
				d2  = dx*dx + dz*dz;
				d = sqrt(d2); 
				if(d < spaceBetweenPolymerCenters)
				{
					test = 0;
				}
				index += PolymerChainLength[j];
			}
		}
		
		TotalPolymerLength = spaceBetweenPolymerCenters * (double)PolymerChainLength[i];
		BodyPosition[k].y = ((double)rand()/(double)RAND_MAX)*(FluidHeight - TotalPolymerLength) +  TotalPolymerLength;
		
		startX = BodyPosition[k].x;
		startY = BodyPosition[k].y;
		startZ = BodyPosition[k].z;
		
		PolymerConnectionA[k] = -1;
		PolymerConnectionB[k] = -1;
		k++;
		
		for(int j = 1; j < PolymerChainLength[i]; j++)
		{
			PolymerConnectionB[k-1] = k;
			PolymerConnectionA[k] = k-1;
			PolymerConnectionB[k] = -1;
			BodyPosition[k].x = startX;
			BodyPosition[k].y = startY - j*spaceBetweenPolymerCenters;
			BodyPosition[k].z = startZ;
			k++;
		}
	}
	
	if(DebugFlag == 1)
	{
		// Printing our polymer chains for debuging.
		k = 0;
		for(int i = 0; i < NumberOfPolymerChains; i++)
		{
			printf("\n ******************* polymer chain %d **********************\n", i);
			for(int j = 0; j < PolymerChainLength[i]; j++)
			{
				printf("PolymerPosition[%d] = (%f, %f, %f) linkA = %d linkB = %d \n", k, BodyPosition[k].x, BodyPosition[k].y, BodyPosition[k].z, PolymerConnectionA[k], PolymerConnectionB[k]);
				printf("PolymerVelocity[%d] = (%f, %f, %f) \n", k, BodyVelocity[k].x, BodyVelocity[k].y, BodyVelocity[k].z);
				k++;
			}
		}
	}
	
	// Shaking the polymers out of their unnatural intial positions.
	polymerShakeUp(BodyPosition, BodyVelocity, BodyForce, PolymerConnectionA, PolymerConnectionB, PolymersConnectionLength, NumberOfPolymers, Drag, Dt, BeakerRadius, FluidHeight);
	
	// Setting intial positions of micro plastics
	for(int i = NumberOfPolymers; i < NumberOfBodies; i++)
	{
		test = 0;
		while(test == 0)
		{
			angle = 2.0*PI*(double)rand()/(double)RAND_MAX;
			BodyPosition[i].x = ((double)rand()/(double)RAND_MAX)*(BeakerRadius * cos(angle));
			BodyPosition[i].y = ((double)rand()/(double)RAND_MAX)*(FluidHeight);
			BodyPosition[i].z = ((double)rand()/(double)RAND_MAX)*(BeakerRadius * sin(angle));
			
			test = 1;
			for(int j = 0; j < i; j++)
			{
				dx = BodyPosition[i].x - BodyPosition[j].x;
				dy = BodyPosition[i].y - BodyPosition[j].y;
				dz = BodyPosition[i].z - BodyPosition[j].z;
				d2  = dx*dx + dy*dy + dz*dz;
				d = sqrt(d2); 
				
				if(d < BodyPosition[i].w + BodyPosition[j].w)
				{
					test = 0;
				}
			}
		}
	}
	
	if(DebugFlag == 1)
	{
		// Printing micro plastics for debugging.
		printf("\n ****************************************** \n");
		for(int i = NumberOfPolymers; i < NumberOfBodies; i++)
		{
			printf(" MicrPlasticPosition[%d] = (%f, %f, %f) \n", i, BodyPosition[i].x, BodyPosition[i].y, BodyPosition[i].z);
		}
	}
		
	printf("\n Initial conditions have been set.\n");
}

void drawPicture()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	// Drawing Polymers
	for(int i = 0; i < NumberOfPolymers; i++)
	{
		glColor3d(PolymerRed, PolymerGreen, PolymerBlue);
		glPushMatrix();
			glTranslatef(BodyPosition[i].x, BodyPosition[i].y, BodyPosition[i].z);
			glutSolidSphere(BodyPosition[i].w/2.0, 30, 30);
		glPopMatrix();
		
		// Drawing polymer connections.
		// Note: there is no need to draw both. If you draw just the one above or below all 
		// connections will be drawn.
		glLineWidth(3.0);
		glColor3d(1.0, 0.0, 0.0);
		glBegin(GL_LINES);
			if(PolymerConnectionA[i] != -1)
			{
				glVertex3f(BodyPosition[i].x, BodyPosition[i].y, BodyPosition[i].z);
				glVertex3f(BodyPosition[PolymerConnectionA[i]].x, BodyPosition[PolymerConnectionA[i]].y, BodyPosition[PolymerConnectionA[i]].z);;
			}
		glEnd();
		
		if(DebugFlag == 1)
		{
			if(PolymerConnectionA[i] == -1)
			{
				glColor3d(0.0, 0.0, 1.0);
				glPushMatrix();
					glTranslatef(BodyPosition[i].x, BodyPosition[i].y, BodyPosition[i].z);
					glutSolidSphere(2.0*BodyPosition[i].w/2.0, 30, 30);
				glPopMatrix();
			}
			if(PolymerConnectionB[i] == -1)
			{
				glColor3d(0.0, 1.0, 1.0);
				glPushMatrix();
					glTranslatef(BodyPosition[i].x, BodyPosition[i].y, BodyPosition[i].z);
					glutSolidSphere(2.0*BodyPosition[i].w/2.0, 30, 30);
				glPopMatrix();
			}
		}
	}
	
	// Drawing Microplastics
	for(int i = NumberOfPolymers; i < NumberOfBodies; i++)
	{
		glColor3d(MicroPlasticRed, MicroPlasticGreen, MicroPlasticBlue);
		glPushMatrix();
			glTranslatef(BodyPosition[i].x, BodyPosition[i].y, BodyPosition[i].z);
			glutSolidSphere(BodyPosition[i].w/2.0, 30, 30);
		glPopMatrix();
	}
	
	// Drawint a red sphere at the origin for reference.
	glColor3d(1.0, 0.0, 0.0);
	glPushMatrix();
		glTranslatef(0, 0, 0);
		glutSolidSphere(10, 30, 30);
	glPopMatrix();
	
	// Drawing the outline of the Beaker.
	if(RadialConfinementViewingAids == 1)
	{
		glLineWidth(1.0);
		float divitions = 60.0;
		float angle = 2.0*PI/divitions;
		
		// Drawing top ring.
		glColor3d(0.0,1.0,0.0);
		for(int i = 0; i < divitions; i++)
		{
			glBegin(GL_LINES);
				glVertex3f(sin(angle*i)*BeakerRadius, FluidHeight, cos(angle*i)*BeakerRadius);
				glVertex3f(sin(angle*(i+1))*BeakerRadius, FluidHeight, cos(angle*(i+1))*BeakerRadius);
			glEnd();
		}

		glColor3d(0.0,1.0,0.0);
		for(int i = 0; i < divitions; i++)
		{
			glBegin(GL_LINES);
				glVertex3f(sin(angle*i)*BeakerRadius, 0.0, cos(angle*i)*BeakerRadius);
				glVertex3f(sin(angle*(i))*BeakerRadius, FluidHeight, cos(angle*(i))*BeakerRadius);
			glEnd();
		}
		
		// Drawing the bottom ring.
		glColor3d(1.0,1.0,1.0);
		for(int i = 0; i < divitions; i++)
		{
			glBegin(GL_LINES);
				glVertex3f(sin(angle*i)*BeakerRadius, 0.0, cos(angle*i)*BeakerRadius);
				glVertex3f(sin(angle*(i+1))*BeakerRadius, 0.0, cos(angle*(i+1))*BeakerRadius);
			glEnd();
		}
	}
	
	// Drawing the stirring.
	if(StirFlag == 1)
	{
		glLineWidth(6.0);
		glColor3d(0.0,1.0,1.0);
		glBegin(GL_LINES);
			glVertex3f(0.0, 0.0, 0.0);
			glVertex3f(BeakerRadius*cos(Theta), 0.0, BeakerRadius*sin(Theta));
		glEnd();
	}
	
	glutSwapBuffers();

	// Captures frames if you are making a movie.
	if(MovieFlag == 1)
	{
		glReadPixels(5, 5, XWindowSize, YWindowSize, GL_RGBA, GL_UNSIGNED_BYTE, Buffer);
		fwrite(Buffer, sizeof(int)*XWindowSize*YWindowSize, 1, MovieFile);
	}
}

void errorCheck(const char *message)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: %s = %s\n", message, cudaGetErrorString(error));
		exit(0);
	}
}

/******************************************************************************
 This function initializes CUDA Rand making it so every thread can have its own set
 of random numbers.
*******************************************************************************/
__global__ void init_curand(unsigned int seed, curandState_t* states) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

/******************************************************************************
 This function controlls all the forces that act on the bodies.
*******************************************************************************/
__global__ void getForces(curandState_t* states, float4 *pos, float4 *vel, float4 *force, int *linkA, int *linkB, float length, int nPolymer, int nPlastics, float beakerRadius, float fluidHeight, float fluidDensity, int stirFlag, float theta, int shakeItUpFlag, float ShakeItUpMag)
{
	int myId, yourId;
	int nBodies;
	float4 forceVector, forceVectorSum;
	float4 velocityVector;
	float4 posMe, posYou;
	float4 velMe;
	float densityMe, massMe;
	
	nBodies = nPolymer + nPlastics;
	myId = threadIdx.x + blockDim.x*blockIdx.x;
    	if(myId < nBodies)
    	{
		posMe.x = pos[myId].x;
		posMe.y = pos[myId].y;
		posMe.z = pos[myId].z;
		posMe.w = pos[myId].w;
		
		velMe.x = vel[myId].x;
		velMe.y = vel[myId].y;
		velMe.z = vel[myId].z;
		velMe.w = vel[myId].w;
		
		//diameterMe = posMe.w;
		densityMe = vel[myId].w;
		massMe = force[myId].w;
		
		forceVectorSum.x = 0.0;
		forceVectorSum.y = 0.0;
		forceVectorSum.z = 0.0;
		
		for(yourId = 0; yourId < nBodies; yourId++)
		{
			posYou.x = pos[yourId].x;
			posYou.y = pos[yourId].y;
			posYou.z = pos[yourId].z;
			posYou.w = pos[yourId].w;
			
			if(yourId != myId) // Making sure you are not working on youself.
			{
				if(myId < nPolymer)
				{
					if(yourId < nPolymer)
					{
						// Polymer-polymer force
						forceVector = getPolymerPolymerForce(posMe, posYou, linkA[myId], linkB[myId], yourId, length, myId);
					}
					else
					{
						// Polymer-microPlastic force
						forceVector = getPolymerMicroPlasticForce(posMe, posYou);
					}
				}
				else
				{
					if(yourId < nPolymer)
					{
						// Polymer-microPlastic force
						forceVector = getPolymerMicroPlasticForce(posMe, posYou);
					}
					else
					{
						// microPlastic-microPlastic force
						forceVector = getMicroPlasticMicroPlasticForce(posMe, posYou);
					}
				}
				
			    	forceVectorSum.x += forceVector.x;
			    	forceVectorSum.y += forceVector.y;
			    	forceVectorSum.z += forceVector.z;
		    	}
		}
		
		// This adds on a gravity pull based on density
		forceVector = getGravityForces(densityMe, massMe, fluidDensity);
		forceVectorSum.x += forceVector.x;
		forceVectorSum.y += forceVector.y;
		forceVectorSum.z += forceVector.z;
		
		// This adds on the forces to keep the bodies in the container.
		forceVector = getContainerForces(posMe, beakerRadius, fluidHeight);
		forceVectorSum.x += forceVector.x;
		forceVectorSum.y += forceVector.y;
		forceVectorSum.z += forceVector.z;
		
		// This adds on the forces caused by stirring.
		if(stirFlag == 1)
		{
			forceVector = getStirringForces(states, myId, posMe, velMe, beakerRadius, fluidHeight, theta);
			forceVectorSum.x += forceVector.x;
			forceVectorSum.y += forceVector.y;
			forceVectorSum.z += forceVector.z;
		}
		
		// This is adds Brownian Motion to the system.
		forceVector = brownian_motion(states, myId);
		forceVectorSum.x += forceVector.x;
		forceVectorSum.y += forceVector.y;
		forceVectorSum.z += forceVector.z;
		
		// This just adds random motion to the system.
		if(shakeItUpFlag == 1)
		{
			velocityVector = shakeItUp(states, myId, ShakeItUpMag);
			vel[myId].x += velocityVector.x;
			vel[myId].y += velocityVector.y;
			vel[myId].z += velocityVector.z;
		}
		
		// Tranfering all the forces to my force function
		force[myId].x = forceVectorSum.x;
		force[myId].y = forceVectorSum.y;
		force[myId].z = forceVectorSum.z;
    	}
}

/******************************************************************************
 This function moves the bodies.
*******************************************************************************/
__global__ void moveBodies(float4 *pos, float4 *vel, float4 *force, float drag, float dt, int n)
{
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	if(id < n)
	{	
		vel[id].x += ((force[id].x-drag*vel[id].x)/force[id].w)*dt;
		vel[id].y += ((force[id].y-drag*vel[id].y)/force[id].w)*dt;
		vel[id].z += ((force[id].z-drag*vel[id].z)/force[id].w)*dt;

		pos[id].x += vel[id].x*dt;
		pos[id].y += vel[id].y*dt;
		pos[id].z += vel[id].z*dt;
	}
}

void nBody()
{
	if(Pause != 1)
	{	
		getForces<<<Grids, Blocks>>>(DevStates, BodyPositionGPU, BodyVelocityGPU, BodyForceGPU, PolymerConnectionAGPU, PolymerConnectionBGPU, PolymersConnectionLength, NumberOfPolymers, NumberOfMicroPlastics, BeakerRadius, FluidHeight, FluidDensity, StirFlag, Theta, ShakeItUpFlag, ShakeItUpMag);
		errorCheck("getForces");
		moveBodies<<<Grids, Blocks>>>(BodyPositionGPU, BodyVelocityGPU, BodyForceGPU, Drag, Dt, NumberOfBodies);
		errorCheck("moveBodies");
        	
        	DrawTimer++;
		if(DrawTimer == DrawRate) 
		{
			cudaMemcpy( BodyPosition, BodyPositionGPU, NumberOfBodies*sizeof(float4), cudaMemcpyDeviceToHost );
			drawPicture();
			//printf("\n Time = %f", RunTime);
			DrawTimer = 0;
		}
		
		PrintTimer++;
		if(PrintRate <= PrintTimer) 
		{
			terminalPrint();
			PrintTimer = 0;
			//printf("\n time = %f", RunTime);
		}
		
		RunTime += Dt; 
		if(TotalRunTime < RunTime)
		{
			printf("\n\n Done\n");
			exit(0);
		}
		
		Theta += StirAngularVelosity*Dt;
		if(2.0*PI < Theta) Theta = 0.0;
	}
}

void terminalPrint()
{
	system("clear");
	//printf("\033[0;34m"); // blue.
	//printf("\033[0;36m"); // cyan
	//printf("\033[0;33m"); // yellow
	//printf("\033[0;31m"); // red
	//printf("\033[0;32m"); // green
	printf("\033[0m"); // back to white.
	
	printf("\n");
	printf("\033[0;33m");
	printf("\n **************************** Simulation Stats ****************************");
	printf("\033[0m");
	
	printf("\n Total run time = %7.2f milliseconds", RunTime);
	
	printf("\033[0;33m");
	printf("\n **************************** Terminal Comands ****************************");
	printf("\033[0m");
	//printf("\n h: Help");
	//printf("\n c: Recenter View");
	printf("\n q: Quit");
	printf("\n m/M: Screenshot/Movie");
	//printf("\n k: Save Current Run");
	printf("\n");
	
	printf("\n Toggles");
	printf("\n r: Run/Pause            - ");
	if(Pause == 0) 
	{
		printf("\033[0;32m");
		printf(BOLD_ON "Simulation Running" BOLD_OFF);
	} 
	else
	{
		printf("\033[0;31m");
		printf(BOLD_ON "Simulation Paused" BOLD_OFF);
	}
	printf("\n v: Orthogonal/Frustum   - ");
	if (ViewFlag == 0) 
	{
		printf("\033[0;36m"); // cyan
		printf(BOLD_ON "Orthogonal" BOLD_OFF); 
	}
	else 
	{
		printf("\033[0;36m"); // cyan
		printf(BOLD_ON "Frustrum" BOLD_OFF);
	}
	printf("\n t: Top/Side view        - ");
	if(TopView == 0) 
	{
		printf("\033[0;36m"); // cyan
		printf(BOLD_ON "Side View" BOLD_OFF);
	}
	else 
	{
		printf("\033[0;36m"); // cyan
		printf(BOLD_ON "Top View" BOLD_OFF);
	}
	printf("\n e: Viewing Aids         - ");
	if(RadialConfinementViewingAids == 0) 
	{
		printf("\033[0;31m");
		printf(BOLD_ON "Viewing Aids Off" BOLD_OFF);
	}
	else 
	{
		printf("\033[0;32m");
		printf(BOLD_ON "Viewing Aids On" BOLD_OFF);
	}
	printf("\n m: Video On/Off         - ");
	if (MovieFlag == 0) 
	{
		printf("\033[0;31m");
		printf(BOLD_ON "Video Recording Off" BOLD_OFF); 
	}
	else 
	{
		printf("\033[0;32m");
		printf(BOLD_ON "Video Recording On" BOLD_OFF);
	}
	printf("\n x: Stirring On/Off      - ");
	if(StirFlag == 0) 
	{
		printf("\033[0;31m");
		printf(BOLD_ON "Stirring Off" BOLD_OFF);
	}
	else 
	{
		printf("\033[0;32m");
		
		printf(BOLD_ON "Stirring..." BOLD_OFF);
	}
	printf("\n c: Shake It Up On/Off   - ");
	if(ShakeItUpFlag == 0) 
	{
		printf("\033[0;31m");
		printf(BOLD_ON "Shake Off" BOLD_OFF);
	}
	else 
	{
		printf("\033[0;32m");
		printf(BOLD_ON "Shake On" BOLD_OFF);
	}
	
	if(TopView == 0) //side view
	{
		printf("\n");
		printf("\n Adjust views");
		printf("\n k/l: Rotate CW/CCW");
		printf("\n a/d: Translate Left/Right");
		printf("\n s/w: Translate Down/Up");
		printf("\n z/Z: Translate Out/In");
		printf("\n f:   Recenter");
		printf("\n");
		printf("\n ********************************************************************");
		printf("\033[0m");
		printf("\n");

	}
	else //top view controls
	{
		printf("\n");
		printf("\n Adjust views");
		printf("\n k/l: Rotate CW/CCW");
		printf("\n a/d: Translate Left/Right");
		printf("\n z/Z: Translate Down/Up");
		printf("\n w/s: Translate Out/In");
		printf("\n f:   Recenter");
		printf("\n");
		printf("\n ********************************************************************");
		printf("\033[0m");
		printf("\n");
	}
}

void setup()
{	
	readSimulationParameters();
	setNumberOfBodies();
	allocateMemory();
	setInitailConditions();
	cudaMemcpy( BodyPositionGPU, BodyPosition, NumberOfBodies*sizeof(float4), cudaMemcpyHostToDevice );
	cudaMemcpy( BodyVelocityGPU, BodyVelocity, NumberOfBodies*sizeof(float4), cudaMemcpyHostToDevice );
	cudaMemcpy( BodyForceGPU, BodyForce, NumberOfBodies*sizeof(float4), cudaMemcpyHostToDevice );
	cudaMemcpy( PolymerConnectionAGPU, PolymerConnectionA, NumberOfPolymers*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( PolymerConnectionBGPU, PolymerConnectionB, NumberOfPolymers*sizeof(int), cudaMemcpyHostToDevice );
	
	// Initialize CURAND
	//unsigned int seed = static_cast<unsigned int>(time(0));
    	init_curand<<<Grids, Blocks>>>(1234, DevStates);
    	errorCheck("init_curand");
	
	cudaSetDevice(0); // Select GPU device 0
    	cudaDeviceSynchronize();
    	errorCheck("cudaSetDevice");
	
	DrawTimer = 0;
	PrintTimer = 0;
	RunTime = 0.0;
	Pause = 1;
	MovieFlag = 0;
	ViewFlag = 1;
	TopView = 1;
	RadialConfinementViewingAids = 1;
	StirFlag = 0;
	ShakeItUpFlag = 0;
	DebugFlag = 0;
	Theta = 0.0;
	StirAngularVelosity = (2.0*PI)/(100.0); // This is 10 revolution per second in milliseconds
	
	CenterOfSimulation.x = 0.0;
	CenterOfSimulation.y = 0.0;
	CenterOfSimulation.z = 0.0;
	CenterOfSimulation.w = 0.0;
	
	AngleOfSimulation.x = 0.0;
	AngleOfSimulation.y = 1.0;
	AngleOfSimulation.z = 0.0;
	AngleOfSimulation.w = 0.0;
	
	terminalPrint();
}

int main(int argc, char** argv)
{
	setup();
	
	XWindowSize = 1000;
	YWindowSize = 1000; 
	Buffer = new int[XWindowSize*YWindowSize];

	// Clip plains
	Near = 0.2;
	Far = FluidHeight*2.0;

	//Direction here your eye is located location
	EyeX = 0.0;
	EyeY = FluidHeight+FluidHeight/3.0;
	EyeZ = 1.0; //BeakerRadius+BeakerRadius/10.0;

	//Where you are looking
	CenterX = 0.0;
	CenterY = 0.0;
	CenterZ = 0.0;

	//Up vector for viewing
	UpX = 0.0;
	UpY = 1.0;
	UpZ = 0.0;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(5,5);
	Window = glutCreateWindow("N Body");
	
	gluLookAt(EyeX, EyeY, EyeZ, CenterX, CenterY, CenterZ, UpX, UpY, UpZ);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, Near, Far);
	glMatrixMode(GL_MODELVIEW);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mymouse);
	glutKeyboardFunc(KeyPressed);
	glutIdleFunc(idle);
	glutMainLoop();
	
	return 0;
}