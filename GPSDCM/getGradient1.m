function g = getGradient1(fun,theta,step)
        %getGradient - Numerical gradient using central differences.
        %   g = getGradient(fun,theta,step) gets numerical gradient of the
        %   function fun evaluated at p-by-1 vector theta. The output g
        %   will be a column vector of size p-by-1. fun is a function
        %   handle to a function that accepts a vector theta and returns a
        %   scalar. Input step is optional.
            
            % 1. Set step size.
            if ( nargin < 3 )
                step = eps^(1/3);
            end
            
            % 2. Initialize output.
            p = length(theta);
            g = zeros(p,1);
            
            parfor i = 1:p
                % 3. Use central differences.
                theta1    = theta;
                theta1(i) = theta1(i) - step;
                
                theta2    = theta;
                theta2(i) = theta2(i) + step;
                
                g(i)      = (fun(theta2) - fun(theta1))/2/step;
            end
            
            % 4. Prevent Inf values in g.
            g = classreg.learning.fsutils.Solver.replaceInf(g,realmax);
        end