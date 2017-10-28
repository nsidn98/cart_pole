import numpy as np
import math
import gym 
import logging
from gym import spaces
from gym.utils import seeding

logger=logging.getLogger(__name__)

class CartPoleEnv(gym.Env):
	metadata={
            'render.modes':['human',rgb_array'],
            'video.frames_per_second':50
            }
    
    def __init__(self):
        self.gravity=9.8
        self.masspole=0.1
        self.masscart=1.0
        self.total_mass=(self.masspole+self.masscart)
        self.length=0.5 #half the pole length
        self.polemass_length=(self.masspole*self.length)
        self.force_mag=10.0
        self.tau=0.02 #seconds between state updates
        
        #angle at which episode is ended
        self.theta_threshold_radians=12*2*math.pi/360 #12 degrees
        self.x_threshold=2.4
        
        #angle limit set to 2*theta_threshold_radians so failing observations are still within bounds
        
        high=np.array([
                    self.x_threshold*2,
                    np.finfo(np.float32).max,
                    self.theta_threshold_radians*2,
                    np.finfo(np.float32).max])
                    
        self.action_space=spaces.Discrete(2)
        self.observation_space=space.Box(-high,high)
        
        self.seed()
        self.viewer=None
        self.state=None
        self.steps_beyond_done=None
        
    def _seed(self,seed=None):
        self.np.random,seed=seeding.np_random(seed)
        return [seed]
    
    def _reset(self):
        self.state=self.np_random.uniform(low=-0.05,high=0.05,size=(4,))
        self.steps_beyond_done=None
        return np.array(self.state)
        
    def _render(self,mode='human',close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer=None
            return
            
        screen_width=600
        screen_height=400
        
        world_width=self.x_threshold*2
        scale=screen_width/world_width
        carty=100 #top of the cart
        polewidth=10.0
        polelen=scale*1.0
        cartwidth=50.0
        cartheight=30.0
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer=rendering.Viewer(screen_width,screen_height)
            l,r,t,b=-cartwidth/2,cartwidth/2,cartheight/2,-cartheight/2
            axleoffset=cartheight/4.0
            cart=rendering.FilledPolygon([(l,b),(l,t),(r,t),(r,b)])
            self.carttrans=rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b=-polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2,
            pole=rendering.FilledPolygon([(l,b),(l,t),(r,t),(r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans=rendering.Transform(translation=(0,axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle=rendering.make_circle(polewidth/2)
	        self.axle.add_attr(self.poletrans)
	        self.axle.add_attr(self.carttrans)
	        self.viewer.add_geom(pole)
            self.axle=rendering.Transform(translation=(0,axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')            

            
    def _step(self,action):
        assert self.action_space.contains(action),"%r (%s) invalid"%(action,type(action))
        state=self.state
        x,x_dot,theta,w=state #w is the angular velocity
        force=self.force_mag if action==1 else -self.force_mag
        costheta=math.cos(theta)
        sintheta=math.sin(theta)
        temp=(force+self.polemass_length*w*w*sintheta)/self.total_mass
        alpha=(self.gravity*sintheta-costheta*temp)/(self.length*(4.0/3.0-self.masspole*costheta*costheta/self.total_mass))
        xacc=temp-self.polemass_length*alpha*costheta/self.total_mass
        x=x+self.tau*x_dot
                            
                            
                          
        
