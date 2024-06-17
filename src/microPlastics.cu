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
using namespace std;

FILE* ffmpeg;

#define PI 3.141592654
#define BLOCK 256

// Globals to be read in from parameter file.
int NumberOfMicroPlastics;
float DensityOfMicroPlastic;
float DiameterOfMicroPlasticMax;
float DiameterOfMicroPlasticMin;

int NumberOfPolymerChains;
int NumberOfPolymers;
int PolymersChainLengthMax;
int PolymersChainLengthMin;
float PolymersConnectionLength;
float DensityOfPolymer;
float DiameterOfPolymer;

float Drag;

float TotalRunTime;
float Dt;
int DrawRate;

// Other Globals
int Pause;
int NumberOfBodies;
float4 *BodyPosition, *BodyVelocity, *BodyForce;
float4 *BodyPositionGPU, *BodyVelocityGPU, *BodyForceGPU;
int *PolymerChainLength;
int *PolymerConnectionA, *PolymerConnectionB;
int *PolymerConnectionAGPU, *PolymerConnectionBGPU;
dim3 Blocks, Grids;
int DrawTimer;
float RunTime;
int* Buffer;
int MovieOn;

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

// Prototyping functions
void readSimulationParameters();
void setNumberOfBodies();
void allocateMemory();
void setInitailConditions();
void drawPicture();
void nBody();
void errorCheck(const char*);
void setup();

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
		data >> DensityOfMicroPlastic;
		
		getline(data,name,'=');
		data >> DiameterOfMicroPlasticMax;
		
		getline(data,name,'=');
		data >> DiameterOfMicroPlasticMin;
		
		getline(data,name,'=');
		data >> NumberOfPolymerChains;
		
		getline(data,name,'=');
		data >> PolymersChainLengthMax;
		
		getline(data,name,'=');
		data >> PolymersChainLengthMin;
		
		getline(data,name,'=');
		data >> PolymersConnectionLength;
		
		getline(data,name,'=');
		data >> DensityOfPolymer;
		
		getline(data,name,'=');
		data >> DiameterOfPolymer;
		
		getline(data,name,'=');
		data >> Drag;
		
		getline(data,name,'=');
		data >> TotalRunTime;
		
		getline(data,name,'=');
		data >> Dt;
		
		getline(data,name,'=');
		data >> DrawRate;
	}
	else
	{
		printf("\nTSU Error could not open simulationSetup file\n");
		exit(0);
	}
	data.close();
	
	printf("\n\n Parameter file has been read");
	printf("\n");
}

void setNumberOfBodies()
{
	time_t t;
	
	PolymerChainLength = (int*)malloc(NumberOfPolymerChains*sizeof(int));
	
	srand((unsigned) time(&t));
	for(int i = 0; i < NumberOfPolymerChains; i++)
	{
		PolymerChainLength[i] = ((float)rand()/(float)RAND_MAX)*(PolymersChainLengthMax - PolymersChainLengthMin) + PolymersChainLengthMin;
		printf("\n PolymerChainLength[%d] = %d", i, PolymerChainLength[i]);
			
	}
	
	NumberOfPolymers = 0;
	for(int i = 0; i < NumberOfPolymerChains; i++)
	{
		NumberOfPolymers += PolymerChainLength[i];	
	}
	
	NumberOfBodies = NumberOfMicroPlastics + NumberOfPolymers;
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
	
	printf("\n\n Memory has been allocated");
	printf("\n");
}

void setInitailConditions()
{
	//float dx, dy, dz, d, d2;
	//int test;
	int startId;
	time_t t;
	srand((unsigned) time(&t));
	
	// Loading velocity, diameter and mass of polymers
	for(int i = 0; i < NumberOfPolymers; i++)
	{
		BodyVelocity[i].x = 0.0;
		BodyVelocity[i].y = 0.0;
		BodyVelocity[i].z = 0.0;
		BodyVelocity[i].w = 0.0;
		
		BodyPosition[i].w = DiameterOfPolymer;	
		BodyForce[i].w = 1.0; //(4.0/3.0)*PI*(BodyPosition[i].w/2.0)*(BodyPosition[i].w/2.0)*(BodyPosition[i].w/2.0);
	}
	
	// Setting velocity, diameter and mass of microplastics
	for(int i = NumberOfPolymers; i < NumberOfBodies; i++)
	{
		BodyVelocity[i].x = 0.0;
		BodyVelocity[i].y = 0.0;
		BodyVelocity[i].z = 0.0;
		BodyVelocity[i].w = 0.0;
		
		BodyPosition[i].w = ((float)rand()/(float)RAND_MAX)*(DiameterOfMicroPlasticMax - DiameterOfMicroPlasticMin) + DiameterOfMicroPlasticMin;
		BodyForce[i].w = 1.0; // (4.0/3.0)*PI*(BodyPosition[i].w/2.0)*(BodyPosition[i].w/2.0)*(BodyPosition[i].w/2.0);	
	}
	
	// Setting intial pos of polymers
	int k = 0;	
	for(int i = 0; i < NumberOfPolymerChains; i++)
	{
		for(int j = 0; j < PolymerChainLength[i]; j++)
		{
			if(j == 0)
			{
				startId = k;
				BodyPosition[k].x = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
				BodyPosition[k].y = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
				BodyPosition[k].z = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
				PolymerConnectionA[k] = -1;
				if(j+1 < PolymerChainLength[i]) PolymerConnectionB[k] = k+1;
				else PolymerConnectionB[k] = -1;
				k++;
			}
			else
			{
				BodyPosition[k].x = BodyPosition[startId].x + j*(PolymersConnectionLength+0.005);
				BodyPosition[k].y = BodyPosition[startId].y;
				BodyPosition[k].z = BodyPosition[startId].z;
				PolymerConnectionA[k] = k-1;
				if(j+1 < PolymerChainLength[i]) PolymerConnectionB[k] = k+1;
				else PolymerConnectionB[k] = -1;
				k++;
			}
		}
	}
	
	for(int i = 0; i < NumberOfPolymers; i++)
	{
		printf("\n %d: A = %d B = %d", i, PolymerConnectionA[i], PolymerConnectionB[i]);
	}
	
	// Setting intial pos of micro plastics
	for(int i = NumberOfPolymers; i < NumberOfBodies; i++)
	{
		BodyPosition[k].x = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
		BodyPosition[k].y = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
		BodyPosition[k].z = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
		k++;
	}

/*		
		test = 0;
		while(test == 0)
		{
			// Get random number between -1 at 1.
			BodyPosition[i].x = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
			BodyPosition[i].y = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
			BodyPosition[i].z = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
			BodyPosition[i].w = ((float)rand()/(float)RAND_MAX)*2.0 - 1.0; 	//MassOfBody;
			test = 1;
			
			for(int j = 0; j < i; j++)
			{
				dx = BodyPosition[i].x-BodyPosition[j].x;
				dy = BodyPosition[i].y-BodyPosition[j].y;
				dz = BodyPosition[i].z-BodyPosition[j].z;
				d2  = dx*dx + dy*dy + dz*dz;
				d = sqrt(d2);
				
				//if(d < DiameterOfBody)
				//{
				//	test = 0;
				//	break;
				//}
			}
			
			if(test == 1)
			{
				BodyVelocity[i].x = 0.0; //VelocityMax*((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
				BodyVelocity[i].y = 0.0; //VelocityMax*((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
				BodyVelocity[i].z = 0.0;  //VelocityMax*((float)rand()/(float)RAND_MAX)*2.0 - 1.0;
				BodyVelocity[i].w = 0.0;
			}
		}
	}
*/
	printf("\n\n Initail conditions have been set.");
	printf("\n");
}

void drawPicture()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	for(int i = 0; i < NumberOfPolymers; i++)
	{
		glColor3d(0.0, 1.0, 0.0);
		glPushMatrix();
			glTranslatef(BodyPosition[i].x, BodyPosition[i].y, BodyPosition[i].z);
			glutSolidSphere(BodyPosition[i].w/2.0, 30, 30);
		glPopMatrix();
	}
	
	for(int i = NumberOfPolymers; i < NumberOfBodies; i++)
	{
		glColor3d(1.0, 1.0, 1.0);
		glPushMatrix();
			glTranslatef(BodyPosition[i].x, BodyPosition[i].y, BodyPosition[i].z);
			glutSolidSphere(BodyPosition[i].w/2.0, 30, 30);
		glPopMatrix();
	}
	
	glutSwapBuffers();
	
	
	if(MovieOn == 1)
	{
		glReadPixels(5, 5, XWindowSize, YWindowSize, GL_RGBA, GL_UNSIGNED_BYTE, Buffer);
		fwrite(Buffer, sizeof(int)*XWindowSize*YWindowSize, 1, ffmpeg);
	}
}

/*
void brownian_motion(float3 *force)
{
	int i,under_normal_curve;
	float mag, angle1, angle2;
	float x,y,normal_hieght,temp;
	
	temp = 4.0*g_drag*DT;
	under_normal_curve = NO;
	
	while(under_normal_curve == NO)
	{
		x = 2.0*1.0*(float)rand()/RAND_MAX - 1.0;
		y = 1.0*(float)rand()/RAND_MAX;
		normal_hieght = 1.0*exp(-x*x/temp);
		if(y <= normal_hieght)
		{
			mag = x;
			under_normal_curve = YES;
		}
	}	
	
	for(i = 0; i < NUMBER_OF_BODIES; i++)
	{
		angle1 = PI*(float)rand()/RAND_MAX;
		angle2 = 2.0*PI*(float)rand()/RAND_MAX;
		force[i].x += mag*sinf(angle1)*cosf(angle2);
		force[i].y += mag*sinf(angle1)*sinf(angle2);
		force[i].z += mag*cosf(angle1);
	}
}
*/
/*
__device__ float4 brownian_motion(float4 p0, other stuff)
{
	get cuda rand working and use the code above to get you started.
}
*/
                                 
__device__ float4 getPolymerPolymerForce(float4 p0, float4 p1, int linkA, int linkB, int yourId, float length, int myId)
{
    float4 f;
    float force;
    float dx = p1.x - p0.x;
    float dy = p1.y - p0.y;
    float dz = p1.z - p0.z;
    float r2 = dx*dx + dy*dy + dz*dz + 0.000001;
    float r = sqrt(r2);
    float penitration = (p0.w + p1.w)/2.0 - r;
    float k1 = 10.1;
    float k2 = 1.1;
    
    force  = 0.0;
    
    if(0.0 < penitration)
    {
    	// PolymerPolymer shell repulsion
    	force  += -k1*penitration;
    }
    else
    {
    	// PolymerPolymer atraction
    	force  += 0.0;
    }
    
    if(yourId == linkA || yourId == linkB)
    {
    	// Polymer chain connection force 
    	force  += -k2*(length - r);
    }
    
    f.x = force*dx/r;
    f.y = force*dy/r;
    f.z = force*dz/r;
    
    return(f);
}

__device__ float4 getPolymerMicroPlasticForce(float4 p0, float4 p1)
{
    float4 f;
    float force;
    float dx = p1.x - p0.x;
    float dy = p1.y - p0.y;
    float dz = p1.z - p0.z;
    float r2 = dx*dx + dy*dy + dz*dz + 0.000001;
    float r = sqrt(r2);
    float penitration = (p0.w + p1.w)/2.0 - r;
    float k = 0.5;
    
    force  = 0.0;
    
    if(0.0 < penitration)
    {
    	// Polymer microPlastic shell repulsion.
    	force  += -k*penitration;
    }
    else if(r < 1.0*(p0.w + p1.w))
    {
    	// Polymer microPlastic actraction
    	force  += 0.1;
    }
    
    f.x = force*dx/r;
    f.y = force*dy/r;
    f.z = force*dz/r;
    
    return(f);
}

__device__ float4 getMicroPlasticMicroPlasticForce(float4 p0, float4 p1)
{
    float4 f;
    float force;
    float dx = p1.x - p0.x;
    float dy = p1.y - p0.y;
    float dz = p1.z - p0.z;
    float r2 = dx*dx + dy*dy + dz*dz + 0.000001;
    float r = sqrt(r2);
    float penitration = (p0.w + p1.w)/2.0 - r;
    float k = 20.2;
    
    force  = 0.0;
    
    if(0.0 < penitration)
    {
    	// MicroPlastic microPlastic shell repulsion.
    	force  += -k*penitration;
    }
    else
    {
    	// MicroPlastic microPlastic actraction
    	force  += 0.0;
    }
    
    f.x = force*dx/r;
    f.y = force*dy/r;
    f.z = force*dz/r;
    
    return(f);
}

__global__ void getForces(float4 *pos, float4 *vel, float4 *force, int *linkA, int *linkB, float length, int nPolymer, int nPlastics)
{
	int myId, yourId;
	int nBodies;
	float4 force_mag, forceSum;
	float4 posMe;
	__shared__ float4 shPos[BLOCK];
	
	nBodies = nPolymer + nPlastics;
	myId = threadIdx.x + blockDim.x*blockIdx.x;
	
    	if(myId < nBodies)
    	{
		forceSum.x = 0.0;
		forceSum.y = 0.0;
		forceSum.z = 0.0;
			
		posMe.x = pos[myId].x;
		posMe.y = pos[myId].y;
		posMe.z = pos[myId].z;
		posMe.w = pos[myId].w;
		
		// Get the Brownian Motion code working and call it here.
		//brownian_motion(posMe, other stuff)
		    
		for(int j = 0; j < gridDim.x; j++)
		{
			yourId = threadIdx.x + blockDim.x*j;
			shPos[threadIdx.x] = pos[yourId];
			__syncthreads();
	   
			#pragma unroll 32
			for(int i = 0; i < blockDim.x; i++)	
			{
				yourId = i + blockDim.x*j;
				if(yourId != myId && yourId < nBodies) 
				{
					if(myId < nPolymer)
					{
						if(yourId < nPolymer)
						{
							// Polymer-polymer force
							force_mag = getPolymerPolymerForce(posMe, shPos[i], linkA[myId], linkB[myId], yourId, length, myId);
						}
						else
						{
							// Polymer-microPlastic force
							force_mag = getPolymerMicroPlasticForce(posMe, shPos[i]);
						}
					}
					else
					{
						if(yourId < nPolymer)
						{
							// Polymer-microPlastic force
							force_mag = getPolymerMicroPlasticForce(posMe, shPos[i]);
						}
						else
						{
							// microPlastic-microPlastic force
							force_mag = getMicroPlasticMicroPlasticForce(posMe, shPos[i]);
						}
					}
					
				    	forceSum.x += force_mag.x;
				    	forceSum.y += force_mag.y;
				    	forceSum.z += force_mag.z;
			    	}
		   	 }
		}
		
		force[myId].x = forceSum.x;
		force[myId].y = forceSum.y;
		force[myId].z = forceSum.z;
    	}
}

__global__ void moveBodies(float4 *pos, float4 *vel, float4 * force, float drag, float dt, int n)
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
	float angle1, angle2;
	float mag = 0.001;
	
	if(Pause != 1)
	{	
		getForces<<<Grids, Blocks>>>(BodyPositionGPU, BodyVelocityGPU, BodyForceGPU, PolymerConnectionAGPU, PolymerConnectionBGPU, PolymersConnectionLength, NumberOfPolymers, NumberOfMicroPlastics);
		moveBodies<<<Grids, Blocks>>>(BodyPositionGPU, BodyVelocityGPU, BodyForceGPU, Drag, Dt, NumberOfBodies);
		
		//*******************************************
		// Pull this out when you get the Brownian Motion stuff working
		cudaMemcpy( BodyVelocity, BodyVelocityGPU, NumberOfBodies*sizeof(float4), cudaMemcpyDeviceToHost );
		
		for(int i = 0; i < NumberOfBodies; i++)
		{
			angle1 = PI*(float)rand()/RAND_MAX;
			angle2 = 2.0*PI*(float)rand()/RAND_MAX;
			BodyVelocity[i].x += mag*sin(angle1)*cos(angle2);
			BodyVelocity[i].y += mag*sin(angle1)*sin(angle2);
			BodyVelocity[i].z += mag*cos(angle1);
		}
		cudaMemcpy( BodyVelocityGPU, BodyVelocity, NumberOfBodies*sizeof(float4), cudaMemcpyHostToDevice );
        	//*******************************************
        	
        	
        	DrawTimer++;
		if(DrawTimer == DrawRate) 
		{
		    cudaMemcpy( BodyPosition, BodyPositionGPU, NumberOfBodies*sizeof(float4), cudaMemcpyDeviceToHost );
			drawPicture();
			//printf("\n Time = %f", RunTime);
			DrawTimer = 0;
		}
		RunTime += Dt; 
		if(TotalRunTime < RunTime)
		{
			printf("\n\n Done\n");
			exit(0);
		}
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
	
	DrawTimer = 0;
	RunTime = 0.0;
	Pause = 1;
}

int main(int argc, char** argv)
{
	setup();
	
	XWindowSize = 1000;
	YWindowSize = 1000; 
	Buffer = new int[XWindowSize*YWindowSize];

	// Clip plains
	Near = 0.2;
	Far = 30.0;

	//Direction here your eye is located location
	EyeX = 0.0;
	EyeY = 0.0;
	EyeZ = 20.0;

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