    def mpc_position1(self):
        try:
            resp = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
            self.states = resp('multirotor', 'ground_plane')
        except rospy.ServiceException:
            print("Can't get model state. Make sure service is available...aborting")
                

        x_states = np.zeros(shape=(8), dtype=float)

        x_states[0] = self.states.pose.position.x
        x_states[1] = self.states.pose.position.y
        x_states[2] = self.states.pose.position.z
        x_states[3] = self.states.twist.linear.x
        x_states[4] = self.states.twist.linear.y
        x_states[5] = self.states.twist.linear.z
        x_states[6] = self.states.pose.orientation.x
        x_states[7] = self.states.pose.orientation.y

        
        # Initial states
        x0 = x_states # Estimated states from GAZEBO
        # x_dot
        # y_dot
        # z_dot
        # vx_dot
        # vy_dot
        # vz_dot
        # roll_dot
        # pitch_dot
        

        # Prediction horizon
        N = 10
        # Define problem
        # Three inputs roll desired, pitch desired, thrust
        u = cp.Variable((3,N))
        # Eight states
        x = cp.Variable((8,N+1))
        #x_init = cp.Parameter(8)

        # Q and R matrices for 
        Q = np.diag([0., 0., 60., 0., 0., 5., 0., 0.])
        R = np.diag([35, 35, 2])

        umin = np.array([-np.pi/6, -np.pi/6, 0.0 ])
        umax = np.array([np.pi/6, np.pi/6, 0.6])
        xmin = np.array([-np.inf,-np.inf,0,-np.inf,-np.inf,-np.inf,
                 -np.pi/6,-np.pi/6])
        xmax = np.array([ np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                  np.pi/6, np.pi/6])


        cost = 0
        constraints = []

        for k in range(N):
            cost += cp.quad_form(x[:,k] - self.xr, Q) + cp.quad_form(u[:,k], R)
            constraints += [x[:,k+1] == self.Ad@x[:,k] + self.Bd@u[:,k]]
            constraints += [xmin <= x[:,k], x[:,k] <= xmax]
            constraints += [umin <= u[:,k], u[:,k] <= umax]
             
        cost += cp.quad_form(x[:,N]-self.xr, Q)
        prob = cp.Problem(cp.Minimize(cost), constraints)

        #nsim = 15
        #for i in range(nsim):
        #    x_init.value = x0
        #    prob.solve()
        #    x0 = Ad.dot(x0) + Bd.dot(u[:,0].value)
        #    x0r = x0
        prob.solve()
        
        rospy.logwarn("X0")
        #rospy.logwarn(prob.solution.primal_vars)
        x = self.Ad.dot(x[:,0].value) + self.Bd.dot(u[:,0].value)
        rospy.logwarn(x.value)
        self.msg.header.stamp = rospy.Time.now()
    
        self.msg.mode = 1
        
        T_nab = ((u[0,2].value+self.gravity)/(15*4)) / (np.cos(x[6])*np.cos(x[7]))# Thrust
        phi_nab = self.gravity * self.xr[6] / T_nab
        theta_nab = self.gravity * self.xr[7] / T_nab
        self.msg.x = phi_nab
        self.msg.y = theta_nab
        self.msg.z = 0.0
        self.msg.F = 0# Thrust
        rospy.logwarn("U-val")
        rospy.logwarn(u.value)

        