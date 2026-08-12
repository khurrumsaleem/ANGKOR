from dataclasses import dataclass, field 
from time import perf_counter

@dataclass
class SolverMetrics:
    case_name:  str     = ""
    use_cmfd:   bool    = False 
    iterations: int     = 0 
    k_history:  list[float] = field(default_factory=list)
    phi_norm_history:   list[float] = field(default_factory=list)
    start_time: float   = 0.0 
    end_time:   float   = 0.0 
    
    def start(self) -> None: 
        self.start_time = perf_counter()
        
    def stop(self) ->  None: 
        self.end_time = perf_counter()
    
    @property
    def elapsed_s(self) -> float: 
        return self.end_time - self.start_time
    
    def record(self, k_eff: float, phi_norm: float) -> None:
        self.iterations += 1
        self.k_history.append(k_eff)
        self.phi_norm_history.append(phi_norm)
        
    def summary(self) -> dict:
        return{
            "case_name"         : self.case_name,
            "use_cmfd"          : self.use_cmfd,
            "iterations"        : self.iterations,
            "final_k_eff"       : self.k_history[-1] if self.k_history else None,
            "final_phi_norm"    : self.phi_norm_history[-1] if self.phi_norm_history else None, 
            "elapsed_s"         : self.elapsed_s,
        }