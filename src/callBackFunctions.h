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

void KeyPressed(unsigned char key, int x, int y)
{	
	if(key == 'q')
	{
		pclose(ffmpeg);
		glutDestroyWindow(Window);
		printf("\nw Good Bye\n");
		exit(0);
	}
	if(key == 'o')
	{
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(-1.0, 1.0, -1.0, 1.0, Near, Far);
		glMatrixMode(GL_MODELVIEW);
		drawPicture();
	}
	if(key == 'f')
	{
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glFrustum(-0.2, 0.2, -0.2, 0.2, Near, Far);
		glMatrixMode(GL_MODELVIEW);
		drawPicture();
	}
	if(key == 'p')
	{
		if(Pause == 1) Pause = 0;
		else Pause = 1;
	}
	if(key == 'm')
	{
		// Setting up the movie buffer.
		const char* cmd = "ffmpeg -r 60 -f rawvideo -pix_fmt rgba -s 1000x1000 -i - "
		              "-threads 0 -preset fast -y -pix_fmt yuv420p -crf 21 -vf vflip output.mp4";
		ffmpeg = popen(cmd, "w");
		//Buffer = new int[XWindowSize*YWindowSize];
		Buffer = (int*)malloc(XWindowSize*YWindowSize*sizeof(int));
		MovieOn = 1;
	}
	if(key == 'M')
	{
		pclose(ffmpeg);
		MovieOn = 0;
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

