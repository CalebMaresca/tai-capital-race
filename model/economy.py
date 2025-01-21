import numpy as np
from model.utils import unconditional_to_conditional

class Economy():
    def __init__(self, rho, beta, alpha, delta, zeta, TFP_0, g_N, g_TAI, unconditional_TAI_probs):
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.delta = delta
        self.zeta = zeta
        self.TFP_0 = TFP_0
        self.g_N = g_N
        self.g_TAI = g_TAI
        self.unconditional_TAI_probs = unconditional_TAI_probs
        self.conditional_TAI_probs = unconditional_to_conditional(unconditional_TAI_probs)
        self.max_TAI_year = len(unconditional_TAI_probs)
        self.g = np.array([g_N] + [g_TAI] * self.max_TAI_year)
    
    def du(self, c):
        """Calculate marginal utility of consumption"""
        if self.rho == 1:
            return 1/c
        else:
            return c**(-self.rho)
    
    def dv_da_t(self, c, r):
        """Derivative of value function with respect to this period's assets"""
        return (1+r) * self.du(c)
    
    def dv_TAI_da_TAI(self, c, w, a_TAI, dv_TAI_da_TAI_next):
        """Derivative of value function with respect to assets in year TAI is invented"""
        return w * self.df_z_da(a_TAI) * self.du(c) + self.beta * dv_TAI_da_TAI_next
    
    def dv_TAI_da_TAI_final(self, c, w, a_TAI):
        """Final period derivative of value function with respect to assets in year TAI is invented"""
        return w * self.df_z_da(a_TAI) / (c**self.rho * (1 - self.beta * (1+self.g_TAI)**(1-self.rho)))
    
    def dv_TAI_da_t_full(self, c, r, w, a, dv_TAI_da_TAI_next):
        """
        Full derivative of value function with respect to this period's assets for year when TAI is invented.
        Full meaning that it includes the effect of an increase in assets on the share of AI labor received.
        """
        return (1+r + w*self.df_z_da(a)) * self.du(c) + self.beta * dv_TAI_da_TAI_next
    
    def dv_N_da_next(self, c, dv_N_next_da_next, dv_TAI_next_da_next, TAI_prob):
        """Derivative of value function with respect to next period's assets for the pre-TAI regime"""
        return - self.du(c) + self.beta * (TAI_prob * dv_TAI_next_da_next + (1 - TAI_prob) * dv_N_next_da_next)
    
    def dv_TAI_da_next(self, c, dv_da_next):
        """Derivative of value function with respect to next period's assets for the TAI regime"""
        return - self.du(c) + self.beta * dv_da_next
    
    def df_z_da(self, a):
        """Derivative of share of AI labor with respect to assets"""
        return self.zeta / a
    
    def rd(self, K, TFP):
        """
        Calculate the interest rate using marginal product of capital
        
        Parameters:
        K: aggregate capital
        TFP: total factor productivity
        
        Returns:
        r: capital rental rate
        """
        return self.alpha * (TFP / K)**(1 - self.alpha) - self.delta

    def r_to_w(self, r, TFP):
        """
        Convert capital rental rate to wage rate using Cobb-Douglas production function
        
        Parameters:
        r: capital rental rate
        
        Returns:
        w: wage rate
        """
        return (1 - self.alpha) * TFP * (self.alpha / (r + self.delta))**(self.alpha / (1 - self.alpha))