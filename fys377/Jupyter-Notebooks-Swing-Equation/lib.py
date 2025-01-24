import numpy as np
from scipy.integrate import odeint

import plotly.graph_objects as go
from plotly.subplots import make_subplots

	
def swing_equation(P, H=6, B=1, D=4, time=25, datapoints=2000, initial_state=True, inf_bus=False, f=None):
	
	M = (2 * H) / (2* np.pi * 50)
	d = D / (2* np.pi * 50)

	if f==None:
		def f(y, t, d, B, M, P):
			return [y[1],
				    ( - d * y[1] + P - B * np.sin(y[0] - y[2]) ) / M, 
				    y[3],
				    ( - d * y[3] - P - B * np.sin(y[2] - y[0]) ) / M]
		
		if inf_bus:
			def f(y, t, d, B, M, P):
		   		return [y[1], ( - d * y[1] + P - B * np.sin(y[0] - y[2]) ) / M, 0, 0]
	
	if initial_state==True:
		initial_phase = -np.dot(np.linalg.pinv(np.array([[-B,B],[B,-B]])), np.array([[P],[-P]]))
		initial_state = [initial_phase[0][0], 0, initial_phase[1][0], 0]

	time = np.linspace(0, time, datapoints)

	res = odeint(f, initial_state, time, args=(d, B, M, P)).T
    	
    # Create a subplot with 1 row and 2 columns
	fig = make_subplots(rows=1, cols=2, subplot_titles=[r'Torque angle', 'Angular velocity'], shared_yaxes=False)

	fig.add_trace(go.Scatter(x=time, y=res[0, :], mode='lines', name='Generator', line=dict(color='black', width=2), legendgroup='1'), row=1, col=1)
	fig.add_trace(go.Scatter(x=time, y=res[2, :], mode='lines', name='Load', line=dict(color='purple', width=2), legendgroup='1'), row=1, col=1)

	fig.update_xaxes(title_text='Time $t$ [s]', row=1, col=1, tickfont=dict(size=16), title_font=dict(size=18))
	fig.update_yaxes(title_text='$\delta$ [rad]', range=[-np.pi/2-0.2, np.pi/2+0.2], tickvals=[-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2], ticktext=[r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$', '0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$'], row=1, col=1, tickfont=dict(size=16), title_font=dict(size=18))

	fig.add_trace(go.Scatter(x=time, y=res[1, :], mode='lines', name='Generator', line=dict(color='black', width=2), legendgroup='2'), row=1, col=2)
	fig.add_trace(go.Scatter(x=time, y=res[3, :], mode='lines', name='Load', line=dict(color='purple', width=2), legendgroup='2'), row=1, col=2)

	fig.update_xaxes(title_text='Time$t$ [s]', row=1, col=2, tickfont=dict(size=16), title_font=dict(size=18))
	fig.update_yaxes(title_text='$\omega$ [rad/s]', range=[-1, 1], row=1, col=2, tickfont=dict(size=16), title_font=dict(size=18))

	arrow1 = go.layout.Annotation(dict(x=24.5, y=res[0, -1], xref="x", yref="y",text="", showarrow=True, axref="x", ayref='y', ax=24.5, ay=0, arrowhead=3, arrowwidth=2, arrowcolor='green'))
	arrow2 = go.layout.Annotation(dict(x=24.5, y=res[2, -1], xref="x", yref="y",text="", showarrow=True, axref="x", ayref='y', ax=24.5, ay=0, arrowhead=3, arrowwidth=2, arrowcolor='green'))

	fig.update_layout(annotations=[arrow1, arrow2])

	l = time[np.argmax(res[0, :] - res[2, :])]
	x_end = [24.5, 24.5, l, l]
	y_end = [res[0, -1], res[2, -1], np.max(res[0, :]), np.min(res[2, :])]
	x_start = [24.5, 24.5, l, l]
	y_start = [0, 0, 0, 0]

	fig.update_layout(annotations=[go.layout.Annotation(dict(x=x0, y=y0, xref="x", yref="y", text="", showarrow=True, axref="x", ayref='y', ax=x1, ay=y1, arrowhead=3, arrowwidth=2, arrowcolor='red',)) for x0,y0,x1,y1 in zip(x_end, y_end, x_start, y_start)])

	fig.add_annotation(text=r"{:.2f}º".format((res[0, -1] - res[2, -1])*180/np.pi), x=22, y=0.0, showarrow=False, font=dict(size=18), xref="x", yref="y")
	fig.add_annotation(text=r"{:.2f}º".format((np.max(res[0, :] - res[2, :]))*180/np.pi), x=l + 4, y=0.0, showarrow=False, font=dict(size=18), xref="x", yref="y")


	fig.update_layout(height=400, width=900, margin=dict(l=0, r=0, t=40, b=0))
	fig.show()
	return fig
