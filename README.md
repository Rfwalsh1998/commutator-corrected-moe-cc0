If gate $g(x)$ is close to 0 or 1, route entirely to one expert (as usual).

If $g(x)$ is intermediate, instead of computing output = $g f_D(x) + (1-g) f_S(x)$, do one of:
(a) Sequential Split: compute $x_1 = f_S(x; \alpha\Delta t)$, then $x_2 = f_D(x_1; \Delta t)$, then $y = f_S(x_2; (1-\alpha)\Delta t)$, where $\alpha = g$ (or $\frac{1}{2}$ for symmetric). Output $y$.
(b) Augmented Blend: compute tentative $y_0 = g f_D(x) + (1-g) f_S(x)$. Also compute $y_c = f_D(x) - f_S(x)$ (or train an expert to predict this difference). Then set $y = y_0 + \frac{1}{2}g(1-g),y_c$ (the factor $\frac{1}{2}g(1-g)$ ensures the correction is largest at mid-gate and vanishes at the ends, scaling the commutator term appropriately).
(c) Lie Algebra Integrator: treat $f_D$ and $f_S$ as defining flows $\dot{x}=D(x)$ and $\dot{x}=S(x)$. Integrate $\dot{x} = g D(x) + (1-g) S(x)$ for time $\Delta t$ (e.g. one step of Euler: $y = x + \Delta t,[,g D(x) + (1-g) S(x),]$ if that’s valid, or a more sophisticated integrator). Use $y$ as output.
