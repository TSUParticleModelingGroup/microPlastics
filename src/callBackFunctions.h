void Display()
{
	drawPicture();
}

void idle()
{
	nBody();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);
}

void orthoganialView()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-BeakerRadius, BeakerRadius, 0.0, FluidHeight, Near, Far); //I think this is what we want, definitely looks 2D
	glMatrixMode(GL_MODELVIEW);
	ViewFlag = 0;
	drawPicture();
}

void fulstrumView()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, Near, Far);
	glMatrixMode(GL_MODELVIEW);
	ViewFlag = 1;
	drawPicture();
}

void topView()
{
	glLoadIdentity();
	glTranslatef(0.0, -CenterY, 0.0);
	glTranslatef(0.0, 0.0, -BeakerRadius);
	glTranslatef(CenterOfSimulation.x, CenterOfSimulation.y, CenterOfSimulation.z);
	glRotatef(90.0, 1.0, 0.0, 0.0);
	glTranslatef(-CenterOfSimulation.x, -CenterOfSimulation.y, -CenterOfSimulation.z);
	glTranslatef(0.0, -FluidHeight+ FluidHeight/10.0, 0.0);
	drawPicture();
}

void sideView()
{
	glLoadIdentity();
	glTranslatef(0.0, -CenterY, 0.0);
	glTranslatef(0.0, -FluidHeight/10.0, -BeakerRadius-BeakerRadius/10.0);
	drawPicture();
}

string getTimeStamp()
{
	// Want to get a time stamp string representing current date/time, so we have a
	// unique name for each video/screenshot taken.
	time_t t = time(0); 
	struct tm * now = localtime( & t );
	int month = now->tm_mon + 1, day = now->tm_mday, year = now->tm_year, 
				curTimeHour = now->tm_hour, curTimeMin = now->tm_min, curTimeSec = now->tm_sec;
	stringstream smonth, sday, syear, stimeHour, stimeMin, stimeSec;
	smonth << month;
	sday << day;
	syear << (year + 1900); // The computer starts counting from the year 1900, so 1900 is year 0. So we fix that.
	stimeHour << curTimeHour;
	stimeMin << curTimeMin;
	stimeSec << curTimeSec;
	string timeStamp;

	if (curTimeMin <= 9)	
		timeStamp = smonth.str() + "-" + sday.str() + "-" + syear.str() + '_' + stimeHour.str() + ".0" + stimeMin.str() + 
					"." + stimeSec.str();
	else			
		timeStamp = smonth.str() + "-" + sday.str() + '-' + syear.str() + "_" + stimeHour.str() + "." + stimeMin.str() +
					"." + stimeSec.str();
	return timeStamp;
}

void screenShot()
{	
	FILE* ScreenShotFile;
	int* buffer;

	const char* cmd = "ffmpeg -loglevel quiet -framerate 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
				"-c:v libx264rgb -threads 0 -preset fast -y -crf 0 -vf vflip output1.mp4";

	ScreenShotFile = popen(cmd, "w");
	buffer = (int*)malloc(XWindowSize*YWindowSize*sizeof(int));
	
	for(int i =0; i < 1; i++)
	{
		drawPicture();
		glReadPixels(5, 5, XWindowSize, YWindowSize, GL_RGBA, GL_UNSIGNED_BYTE, buffer);
		fwrite(buffer, sizeof(int)*XWindowSize*YWindowSize, 1, ScreenShotFile);
	}
	
	pclose(ScreenShotFile);
	free(buffer);

	string ts = getTimeStamp(); // Only storing in a separate variable for debugging purposes.
	string s = "ffmpeg -loglevel quiet -i output1.mp4 -qscale:v 1 -qmin 1 -qmax 1 " + ts + ".jpeg";
	// Convert back to a C-style string.
	const char *ccx = s.c_str();
	system(ccx);
	system("rm output1.mp4");
	printf("\nScreenshot Captured: \n");
	cout << "Saved as " << ts << ".jpeg" << endl;
	
	system("mv *.jpeg Stills/");
}

void movieOn()
{
	string ts = getTimeStamp();
	ts.append(".mp4");

	string baseCommand = "ffmpeg -loglevel quiet -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
				"-c:v libx264rgb -threads 0 -preset fast -y -pix_fmt yuv420p -crf 0 -vf vflip ";

	string z = baseCommand + ts;
	const char *ccx = z.c_str();
	MovieFile = popen(ccx, "w");
	Buffer = (int*)malloc(XWindowSize*YWindowSize*sizeof(int));
}

void movieOff()
{
	pclose(MovieFile);
	free(Buffer);
	system("mv *.mp4 Videos/");
}


void KeyPressed(unsigned char key, int x, int y)
{	
	cudaMemcpy( BodyPosition, BodyPositionGPU, NumberOfBodies*sizeof(float4), cudaMemcpyDeviceToHost );
	float dAngle = 0.01;
	float dx = 100;
	float dy = 100;
	float dz = 100;
	float temp;
	

	if(key == 'Q')
	{
		if(MovieFlag == 1) 
		{
			movieOff();
		}
		glutDestroyWindow(Window);
		printf("\n Good Bye \n");
		exit(0);
	}
	
	if(key == 'v') // Orthoganal/Fulstrium view
	{
		if(ViewFlag == 0) 
		{
			ViewFlag = 1;
			fulstrumView();
		}
		else 
		{
			ViewFlag = 0;
			orthoganialView();
		}
		drawPicture();
		terminalPrint();
	}
	
	if(key == 'r')
	{
		if(Pause == 1) 
		{
			Pause = 0;
			terminalPrint();
		}
		else 
		{
			Pause = 1;
			terminalPrint();
		}
	}
	if(key == 'm')
	{
		if(MovieFlag == 0) 
		{
			MovieFlag = 1;
			movieOn();
			terminalPrint();
		}
		else 
		{
			MovieFlag = 0;
			movieOff();
			terminalPrint();
		}	
	}
	if(key == 'q')
	{
		screenShot();
	}
	
	// Rotate clockwise on the y-axis
	if(key ==  'l')
	{
		/* //in case there is another way
		glRotatef(dAngle, 0.0,1.0,0.0);
			AngleOfSimulation.y += dAngle;
			drawPicture();
		*/
		for(int i = 0; i < NumberOfBodies; i++)
		{
			BodyPosition[i].x -= CenterOfSimulation.x;
			BodyPosition[i].y -= CenterOfSimulation.y;
			BodyPosition[i].z -= CenterOfSimulation.z;
			temp =  cos(-dAngle)*BodyPosition[i].x + sin(-dAngle)*BodyPosition[i].z;
			BodyPosition[i].z  = -sin(-dAngle)*BodyPosition[i].x + cos(-dAngle)*BodyPosition[i].z;
			BodyPosition[i].x  = temp;
			BodyPosition[i].x += CenterOfSimulation.x;
			BodyPosition[i].y += CenterOfSimulation.y;
			BodyPosition[i].z += CenterOfSimulation.z;
		}
		drawPicture();
		AngleOfSimulation.y -= dAngle;

	}
	if(key == 'k')  // Rotate counter clockwise on the y-axis, us glRotate
	{
		/* //in case there is another way
		glRotatef(-dAngle, 0.0,1.0,0.0);
			AngleOfSimulation.y -= dAngle;
			drawPicture();
		*/
		for(int i = 0; i < NumberOfBodies; i++)
		{
			BodyPosition[i].x -= CenterOfSimulation.x;
			BodyPosition[i].y -= CenterOfSimulation.y;
			BodyPosition[i].z -= CenterOfSimulation.z;
			temp = cos(dAngle)*BodyPosition[i].x + sin(dAngle)*BodyPosition[i].z;
			BodyPosition[i].z  = -sin(dAngle)*BodyPosition[i].x + cos(dAngle)*BodyPosition[i].z;
			BodyPosition[i].x  = temp;
			BodyPosition[i].x += CenterOfSimulation.x;
			BodyPosition[i].y += CenterOfSimulation.y;
			BodyPosition[i].z += CenterOfSimulation.z;
		}
		drawPicture();
		AngleOfSimulation.y += dAngle;
	}
	if(key == 'a')  // Translate left on the x-axis
	{
		glTranslatef(dx, 0.0, 0.0);
			DistanceFromCenter.x += dx;
			drawPicture();
	}
	if(key == 'd')  // Translate right on the x-axis
	{
		glTranslatef(-dx, 0.0, 0.0);
			DistanceFromCenter.x += -dx;
			drawPicture();
	}
	if(key == 's')  // Translate down on the y-axis
	{
			glTranslatef(0.0, dy, 0.0);
			DistanceFromCenter.y += dy;
			drawPicture();
	}
	if(key == 'w')  // Translate up on the y-axis
	{
		glTranslatef(0.0, -dy, 0.0);
			DistanceFromCenter.y += -dy;
			drawPicture();
	}
	if(key == 'z')  // Translate out on the z-axis
	{
		glTranslatef(0.0, 0.0, -dz);
			DistanceFromCenter.z += -dz;
			drawPicture();
	}
	if(key == 'Z')  // Translate in on the z-axis
	{
		glTranslatef(0.0, 0.0, dz);
			DistanceFromCenter.z += dz;
			drawPicture();
	}
	if(key == 'f')  // Recenter
	{
		glTranslatef(-DistanceFromCenter.x, -DistanceFromCenter.y, -DistanceFromCenter.z);
			DistanceFromCenter.x = 0.0;
			DistanceFromCenter.y = 0.0;
			DistanceFromCenter.z = 0.0;
			drawPicture();
	}
	if(key == 'e')//toggle rings
	{
		if(RadialConfinementViewingAids == 0)
		{
			RadialConfinementViewingAids = 1;
			terminalPrint();
			drawPicture();
		}
		else
		{
			RadialConfinementViewingAids = 0;
			terminalPrint();
			drawPicture();
		}

	}
	if(key == 't')
	{
		if(TopView == 0) //looking at the side currently, change to top
		{
			TopView = 1;
			topView();
			terminalPrint();
			drawPicture();

		}
		else
		{
			TopView = 0;
			sideView();
			terminalPrint();
			drawPicture();
		}
	}
	if(key =='x')//stir
	{
		if(StirFlag == 0)
		{
			printf("How fast would you like to Stir?\n");
			printf("Enter in Rev/s and we will convert it to milliseconds.\n");
			scanf("%f", &StirAngularVelosity);
			StirAngularVelosity = ((2.0*PI)/1000.0)*StirAngularVelosity;
			StirFlag = 1;
			terminalPrint();
			drawPicture();
		}
		else
		{
			StirFlag = 0;
			terminalPrint();
			drawPicture();
		}


	}
	if(key == 'c')//shake
	{
		if(ShakeItUpFlag == 0)
		{
			ShakeItUpFlag = 1;
			printf("Enter the magnitude of the shake\n");
			scanf("%f", &ShakeItUpMag);
			terminalPrint();
			drawPicture();
		}
		else
		{
			ShakeItUpFlag = 0;
			terminalPrint();
			drawPicture();
		}

	}
	
	cudaMemcpy( BodyPositionGPU, BodyPosition, NumberOfBodies*sizeof(float4), cudaMemcpyHostToDevice );
}

void mymouse(int button, int state, int x, int y)
{	
	//float myX, myY, myZ;
	//int index = -1;
	
	if(state == GLUT_DOWN)
	{
		if(button == GLUT_LEFT_BUTTON)
		{
			//printf("\n Left mouse button down");
			//printf("\n mouse x = %d mouse y = %d\n", x, y);
			//myX = (2.0*x/XWindowSize - 1.0)*RadiusOfCavity;
			//myY = (-2.0*y/YWindowSize + 1.0)*RadiusOfCavity;
		}
		else
		{
			//printf("\nRight mouse button down");
			//printf("\nmouse x = %d mouse y = %d\n", x, y);
			//myX = (2.0*x/XWindowSize - 1.0)*RadiusOfAtria;
			//myY = (-2.0*y/YWindowSize + 1.0)*RadiusOfAtria;
		}
		//printf("\nSNx = %f SNy = %f SNz = %f\n", BodyPositionPosition[0].x, BodyPositionPosition[0].y, BodyPositionPosition[0].z);
	}
}

