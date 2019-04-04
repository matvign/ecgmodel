% MATLAB implementation of dynamic ECG model

% The model is plotted in 3d space.
% The ECG trajectory is plotted on a unit circle. There are five events
% plotted on the unit circle starting at 12 o'clock.
% These events are represented by PQRST.
% The trajectory gets pulled up or down when it approaches these events.

events = [-pi/3, -pi/12, 0, pi/12, pi/2];  % event angles for the limit circle
a = [1.2, -5, 30, -7.5, 0.75];             % values of a for each event
b = [0.25, 0.1, 0.1, 0.1, 0.4];            % values of b for each event
w = 2*pi;                              % value for angular velocity
tspan = [-1 1];                        % plot for these time values
y0 = [-1; 0; 0];                       % vector of initial values

% ode15s(@(t,z) odefcn(...), tspan, y0, options)
% https://au.mathworks.com/help/matlab/ref/ode15s.html
[time, y_out] = ode15s(@(t,y) odefcn(t,y,a,b,w,events), tspan, y0);

plot(time, y_out(:,3), 'k'),
    xlabel('time (s)'),
    ylabel('mV'),
    title('Synthetic ECG');


function dy = odefcn(T, Y, a, b, w, events)
    dy = zeros(3, 1);
    x = Y(1);
    y = Y(2);
    z = Y(3);
    
    theta = atan2(y, x);
    alpha = 1 - sqrt(x^2 + y^2);    
    dy(1) = alpha*x - w*y;
    dy(2) = alpha*y + w*x;

    dy(3) = -(z - 0);
    for i = 1:5
        dy(3) = dy(3) -...
            a(i) * (theta - events(i)) * exp(-(theta - events(i))^2/(2*b(i)^2));
    end
end

function z_0 = get_z0(t, freq)
    % unused function
    % freq is respiratory frequency
    z_0 = 0.15 * sin(2*pi*freq*t);
end

function dtime = get_dtime(freq)
    % unused function
    % freq = sampling frequency
    dtime = 1/freq;
end