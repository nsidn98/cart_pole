#include
#define sqr(x) ((x)*(x))
#define W_INIT 0.0
#define NUM_BOXES 162
extern int RND_SEED;
static float ALPHA=0.5;  //learning rate
static float GAMMA=0.999; //discount factor
static float BETA=0.0;   //magnitude of noise added

static float q_val[NUM_BOXES][2];  //state action values
static first_time=1;
static int current_action,prev_action;
static int current_state,prev_state;
static char rcs_controller_id[] = "$Id: q.c,v 1.1.1.1 1995/02/10 21:49:24 finton Exp $";
int get_box(float x,float velocity,float theta,float ang_velocity);

/*
	get_action returns either 0 or 1 
	it accepts five black box inputs in which first four are 
	system variables and the last is a reinforcement signal
	where reinforcement signal is the result of previous statr and action
*/	
int get_action(float x,float velocity,float theta,float ang_velocity,float reinf)
{
	int i,j;			
	float predicted_value;	// max_(b) Q(t,ss,b)
	int state_box(float x,float velocity,float theta,float ang_velocity);
	double rnd(double,double);
	void srandom(int);
	void reset_controller(void);	//reset state/ction before new trial

	if(first_time){
		first_time=0;
		reset_controller(); 	//set the state and action to null values

		for(i=0;i<NUM_BOXES;i++)
			for(j=0;j<2;j++)
				q_val[i][j]=W_INIT;  	//initialize the state values to zero  

		printf("Controller:%s\n",rcs_controller_id);
		printf("...setting learning parameter ALPHA to %.4f.\n",ALPHA);
		printf("... setting noise parameter BETA to %.4f.\n", BETA);
        printf("... setting discount parameter GAMMA to %.4f.\n", GAMMA);
      	printf("... random RND_SEED is %d.\n", RND_SEED);
      	srandom(RND_SEED);	
	}

	prev_state=current_state;
	prev_action=current_action;
	current_state=get_box(x,velocity,theta,ang_velocity);

	if(prev_action!=-1) //update except forthe first action
	{
		if(current_state==-1)
			predicted_value=0.0;//failure state has Q-value of 0 since the value won't be updated
		else if(q_val[current_state][0]<=q_val[current_state][1])
			predicted_value=q_val[current_state][1];
		else
			predicted_value=q_val[current_state][0];

		q_val[prev_state][prev_action]+=ALPHA*(reinf+GAMMA*predicted_value-q_val[prev_state][prev_action]);

	}	

	//to determine best action
	if(q_val[current_state][0]+rnd(-BETA,BETA)<=q_val[current_state][1])
		current_action=1;
	else
		current_action=0;

	return current_action;

}


double rnd(double low_bound,double hi_bound) //it scales the output to the range [low_bound,hi_bound]
{
	long random(void);			//random number generator
	double highest=(double)((1 << 31) -1);
	//if rand_max is not defined then try((1<<31)-1)
	return (random()/highest)*(hi_bound-low_bound)+low_bound;
}

void reset_controller(void)
{
	current_state=prev_state=0;
	current_action=prev_action=-1;
}

/*following sub-routine was written by Rich Sutton and Chuck Anderson with translation 
   from FORTRAN to C by Claude Sammut  */

#define one_degree 0.0174532	/* 2pi/360 */
#define six_degrees 0.1047192
#define twelve_degrees 0.2094384
#define fifty_degrees 0.87266

int get_box(float x,float velocity,float theta,float ang_velocity)
{
	int box=0;

	if(x<2.4||x>2.4||theta<-twelve_degrees||theta>twelve_degrees)
		return(-1);			// return signal failure if it goes outside the limits set

	if(x<-0.8)						box=0;
	else if (x<0.8)					box=1;
	else 							box=2;

	if(velocity<-0.5)				;
	else if(velocity<0.5)			box+=3;
	else 							box+=6;

	if (theta < -six_degrees) 	    ;
  	else if (theta < -one_degree)   box += 9;
  	else if (theta < 0) 		    box += 18;
  	else if (theta < one_degree) 	box += 27;
  	else if (theta < six_degrees)   box += 36;
  	else	    			       	box += 45;

  	if(ang_velocity<-fifty_degrees)	;
  	else if (ang_velocity<fifty_degrees) box+=54;
  	else									box+=108;


  	return box;

}


/*The system must must learn to correlate its state in the environment       |
|  with the future reinforcements it will see for each of its candidate       |
|  actions.  Therefore, the system must determine its state from the          |
|  signal values it gets from the environment.  For the current               |
|  demonstration, the system does no feature extraction;  instead, it         |
|  determines its state according to a built-in look-up table.  This          |
|  table is the "boxes" state representation of Barto, Sutton, and            |
|  Anderson, described in their paper, "Neuronlike Adaptive Elements That     |
|  Solve Difficult Learning Control Problems," IEEE Trans. Syst., Man,        |
|  Cybern., Vol. SMC-13, pp. 834-846, Sep.- Oct. 1983.*/
