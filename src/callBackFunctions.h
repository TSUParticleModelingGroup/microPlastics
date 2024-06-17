void Display(void);
void idle();
void reshape(int, int);
void orthoganialView();
void fulstrumView();
void KeyPressed(unsigned char, int, int);
void mymouse(int, int, int, int);
string getTimeStamp();
void movieOn();
void movieOff();
void screenShot();
void helpMenu();



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
	glOrtho(-1.0, 1.0, -1.0, 1.0, Near, Far);
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

void KeyPressed(unsigned char key, int x, int y)//movekeyPressed functions over as well
{	
	if(key == 'h')  // Help menu
	{
		helpMenu();
	}
	if(key == 'q')
	{
		pclose(ffmpeg);
		glutDestroyWindow(Window);
		printf("\nw Good Bye\n");
		exit(0);
	}
	if(key == 'r')  // Run toggle
	{
		if(Pause == 0) Pause = 1;
		else Pause = 0;
		terminalPrint();
	}
	if(key == 'v') // Orthoganal/Frustum view toggle
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
	if(key == 'S')  // Screenshot
	{	
		screenShot();
		terminalPrint();
	}
	if(key == 'm')  // Movie on
	{
		if(MovieFlag == 0) 
		{
			MovieFlag = 1;
			movieOn();
		}
		else 
		{
			MovieFlag = 0;
			movieOff();
		}
		terminalPrint();
	}
	if(key == 'c')  // Recenter image
	{
		CenterX = 0.0;
		CenterY = 0.0;
		CenterZ = 0.0;
		drawPicture();
		terminalPrint();
	}
}

void mymouse(int button, int state, int x, int y)
{	
	//float myX, myY, myZ;
	//int index = -1;
	
	if(state == GLUT_DOWN)
	{
		if(button == GLUT_LEFT_BUTTON)
		{
			printf("\n Left mouse button down");
			printf("\n mouse x = %d mouse y = %d\n", x, y);
			//myX = (2.0*x/XWindowSize - 1.0)*RadiusOfCavity;
			//myY = (-2.0*y/YWindowSize + 1.0)*RadiusOfCavity;
		}
		else
		{
			printf("\nRight mouse button down");
			printf("\nmouse x = %d mouse y = %d\n", x, y);
			//myX = (2.0*x/XWindowSize - 1.0)*RadiusOfAtria;
			//myY = (-2.0*y/YWindowSize + 1.0)*RadiusOfAtria;
		}
		//printf("\nSNx = %f SNy = %f SNz = %f\n", NodePosition[0].x, NodePosition[0].y, NodePosition[0].z);
	}
}

string getTimeStamp()//for ss/videos
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

void movieOn()
{
	string ts = getTimeStamp();
	ts.append(".mp4");

	// Setting up the movie buffer.
	const char* cmd = "ffmpeg -loglevel quiet -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
		      "-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output.mp4";

	string baseCommand = "ffmpeg -loglevel quiet -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
				"-c:v libx264rgb -threads 0 -preset fast -y -pix_fmt yuv420p -crf 0 -vf vflip ";

	string z = baseCommand + ts;

	const char *ccx = z.c_str();
	MovieFile = popen(ccx, "w");
	//Buffer = new int[XWindowSize*YWindowSize];
	Buffer = (int*)malloc(XWindowSize*YWindowSize*sizeof(int));
	MovieOn = 1;
}

void movieOff()
{
	if(MovieOn == 1) 
	{
		pclose(MovieFile);
	}
	free(Buffer);
	MovieOn = 0;
}


void screenShot()//yes
{	
	int pauseFlag;
	FILE* ScreenShotFile;
	int* buffer;

	const char* cmd = "ffmpeg -loglevel quiet -framerate 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
				"-c:v libx264rgb -threads 0 -preset fast -y -crf 0 -vf vflip output1.mp4";
	//const char* cmd = "ffmpeg -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
	//              "-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output1.mp4";
	ScreenShotFile = popen(cmd, "w");
	buffer = (int*)malloc(XWindowSize*YWindowSize*sizeof(int));
	
	if(Pause == 0) 
	{
		Pause = 1;
		pauseFlag = 0;
	}
	else
	{
		pauseFlag = 1;
	}
	
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

	
	//system("ffmpeg -i output1.mp4 screenShot.jpeg");
	//system("rm output1.mp4");
	
	Pause = pauseFlag;
	//ffmpeg -i output1.mp4 output_%03d.jpeg
}

void helpMenu()//yes, but be sure to adjust for mp
{
	system("clear");
	//Pause = 1;
	printf("\n The simulation is paused.");
	printf("\n");
	printf("\n h: Help");
	printf("\n q: Quit");
	printf("\n r: Run/Pause (Toggle)");
	//printf("\n g: View front half only/View full image (Toggle)");
	printf("\n v: Orthogonal/Frustum projection (Toggle)");
	printf("\n");
	printf("\n m: Movie on/Movie off (Toggle)");
	printf("\n S: Screenshot");
	printf("\n");
	printf("\n c: Recenter image");
	printf("\n w: Counterclockwise rotation x-axis");
	printf("\n s: Clockwise rotation x-axis");
	printf("\n d: Counterclockwise rotation y-axis");
	printf("\n a: Clockwise rotation y-axis");
	printf("\n z: Counterclockwise rotation z-axis");
	printf("\n Z: Clockwise rotation z-axis");
	printf("\n e: Zoom in");
	printf("\n E: Zoom out");
	printf("\n");
}
