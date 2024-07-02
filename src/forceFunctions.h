__device__ float4 brownian_motion(curandState_t*, int);
__device__ float4 shakeItUp(curandState_t*, int);
__device__ float4 getPolymerPolymerForce(float4, float4, int, int, int, float, int);
__device__ float4 getPolymerMicroPlasticForce(float4, float4);
__device__ float4 getMicroPlasticMicroPlasticForce(float4, float4);
__device__ float4 getGravityForces(float, float, float);
__device__ float4 getContainerForces(float4, float , float);
__device__ float4 getStirringForces(curandState_t*, int, float4, float4, float, float, float);

/******************************************************************************
 This is the Brownian Motion function.
 Place any comments and papers you used to get parameters for this function here.
 The above commented out function is one I ased to get Brownian Motion in another project.
*******************************************************************************/
__device__ float4 brownian_motion(curandState_t* states, int id)
{
	float mag = 100.0;
	float4 f;
	float randx = mag*(curand_uniform(&states[id])*2.0 - 1.0);
        float randy = mag*(curand_uniform(&states[id])*2.0 - 1.0);
        float randz = mag*(curand_uniform(&states[id])*2.0 - 1.0);
        
        f.x = randx;
        f.y = randy;
        f.z = randz;
	
	return(f);
}

/******************************************************************************
 This function just shakes the whole system up
*******************************************************************************/
__device__ float4 shakeItUp(curandState_t* states, int id)
{
	float mag = 10.0;
	float4 v;
	float randx = mag*(curand_uniform(&states[id])*2.0 - 1.0);
        float randy = mag*(curand_uniform(&states[id])*2.0 - 1.0);
        float randz = mag*(curand_uniform(&states[id])*2.0 - 1.0);
        
        v.x = randx;
        v.y = randy;
        v.z = randz;
	
	return(v);
}

/******************************************************************************
 This is the Polymer to Polymer interaction function.
 Place any comments and papers you used to get parameters for this function here.
 
*******************************************************************************/                                 
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
    float k1 = 100.0;
    float k2 = 100.0;
    
    force  = 0.0;
    
    if(0.0 < penitration)
    {
    	// PolymerPolymer shell repulsion
    	force  += -k1*penitration*penitration;
    }
    else
    {
    	// PolymerPolymer atraction
    	force  += 0.0;
    }
    
    // Polymer chain connection forces.
    if(linkA != -1 && yourId == linkA)
    { 
    	force  += -k2*(length - r);
    }
    if(linkB != -1 && yourId == linkB)
    { 
    	force  += -k2*(length - r);
    }
    
    f.x = force*dx/r;
    f.y = force*dy/r;
    f.z = force*dz/r;
    
    return(f);
}

/******************************************************************************
 This is the Polymer to micro-plastic interaction function.
 Place any comments and papers you used to get parameters for this function here.
*******************************************************************************/
__device__ float4 getPolymerMicroPlasticForce(float4 p0, float4 p1)
{
    float4 f;
    float force;
    float dx = p1.x - p0.x;
    float dy = p1.y - p0.y;
    float dz = p1.z - p0.z;
    float r2 = dx*dx + dy*dy + dz*dz + 0.000001;
    float r = sqrt(r2);
    float G = 100.0;
    float penitration = (p0.w + p1.w)/2.0 - r;
    float k = 100.0;
    
    force  = 0.0;
 
    if(0.0 < penitration)
    {
    	// Polymer microPlastic shell repulsion.
    	force  += -k*penitration*penitration;
    }
    else
    {
    	// Polymer microPlastic actraction
    	force += G*(p0.w*p1.w)/r2;
    	//force += 0.0;
    	//printf("\n force = %f", force);
    }
    
    f.x = force*dx/r;
    f.y = force*dy/r;
    f.z = force*dz/r;
    return(f);
}

/******************************************************************************
 This is the micro-plasic to micro-plastic interaction function.
 Place any comments and papers you used to get parameters for this function here.
 Self-Assembled Plasmonic Nanoparticle Clusters: 
 https://www.science.org/doi/10.1126/science.1187949#editor-abstract
*******************************************************************************/
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
    float k = 100.0;
    
    force  = 0.0;
    
    if(0.0 < penitration)
    {
    	// MicroPlastic microPlastic shell repulsion.
    	force  += -k*penitration*penitration;
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

/******************************************************************************
 This is the gravity function add complexity at will.
*******************************************************************************/
__device__ float4 getGravityForces(float density, float mass, float fluidDensity)
{
	float4 f;
	float G = 9.81; // When you take meters/second^2 to micrometers/millisecond^2 everything cancels out so you get 9.81
	f.x = 0.0;
	f.z = 0.0;
	
	// May want to do something more accurate. I just made a linear function that  pulled stuff dowm if its density is
	// greater than the fluid pushed it up if its density is less than the fluid.
	f.y = -G*mass*(density - fluidDensity);
	
	return(f);
}

/******************************************************************************
 This function keeps the bodies in the container.
*******************************************************************************/
__device__ float4 getContainerForces(float4 posMe, float beakerRadius, float fluidHeight)
{
	float4 f;
	float force;
	float r2 = posMe.x*posMe.x + posMe.z*posMe.z;
	float r = sqrt(r2);
	float k = 100.0;
	
	f.x = 0.0;
	f.y = 0.0;
	f.z = 0.0;
	
	if(beakerRadius < r)
	{
		force = k*(beakerRadius - r);
		f.x = force*posMe.x/r;
		f.z = force*posMe.z/r;
	}
	
	if(fluidHeight < posMe.y)
	{
		force = k*(fluidHeight - posMe.y);
		f.y = force;
	}
	else if(posMe.y < 0.0)
	{
		force = -k*(posMe.y);
		f.y = force;
	}
	
	return(f);
}

/******************************************************************************
 This is the stirring function add complexity at will.
*******************************************************************************/
__device__ float4 getStirringForces(curandState_t* states, int id, float4 posMe, float4 velMe, float beakerRadius, float fluidHeight, float theta)
{
	float4 f;
	float angle;
	float magRand = 10000.0;
	float centerMag;
	//float temp;
	float magStir = 200.0; 
	//float mag2 = 10.0;
	float r2 = posMe.x*posMe.x + posMe.z*posMe.z;
	float r = sqrt(r2);
	float range = PI/6.0;
	
	float randx = magRand*(curand_uniform(&states[id])*2.0 - 1.0);
        float randy = magRand*(curand_uniform(&states[id])*2.0 - 1.0);
        float randz = magRand*(curand_uniform(&states[id])*2.0 - 1.0);
	
	f.x = 0.0;
	f.y = 0.0;
	f.z = 0.0;
	
	if(0.0 < r)
	{
		// This gives a radial motion
		//f.x = mag1*(-posMe.z/r);
		//f.z = mag1*(posMe.x/r);
		
		// This gives a pulling down in the center and up on the sides.
		//f.y = mag2*(r*2.0/beakerRadius - 1.0);
		
		// This is suposed to move it in from the top and out on the bottom.
		//temp = 10.0*(1.0 - posMe.y/fluidHeight); //mag2*(-r*2.0/beakerRadius + 1.0);
		//f.x = temp*(posMe.x/r);
		//f.z = temp*(posMe.z/r);
		
		angle = atan(posMe.z/posMe.x);
		if(0.0 < posMe.x)
		{
			if(0.0 < posMe.z)
			{
				angle += 0.0;
			}
			else
			{
				angle += 2.0*PI;
			}
		}
		else
		{
			if(0.0 < posMe.z)
			{
				angle += PI;
			}
			else
			{
				angle += PI;
			}
		}
		
		if(0.0 < (angle - theta) && (angle - theta) < range || (2*PI - (theta - angle)) < range)
		{
			centerMag = -(r/beakerRadius - 1.0)*(r/beakerRadius - 1.0) + 1.0; // This makes it full in the middle and die off on the ends,
			f.x = randx + centerMag*magStir*(-posMe.z/r);
			f.y = randy;
			f.z = randz + centerMag*magStir*(posMe.x/r);
		}
	}
	
	return(f);
}

// This is an example of brownian motion I used in another project.
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

