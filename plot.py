# ST_CA_TA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_csv('BTP_4.csv')

features_input = np.array(['Surface Tension','Contact Angle','viscosity'])
features_output = np.array(['TA','AW','TL','FL'])

X = df[features_input]
Y = df[features_output]

bootstrap = [False]
n_estimators = [12]
max_depth = [20]
min_samples_split = [3]
min_samples_leaf = [1]

rf = RandomForestRegressor(bootstrap=False,n_estimators=12,max_depth=20,min_samples_leaf=1,min_samples_split=3)

rf.fit(X,Y)

x = np.array(X['Surface Tension'])
y = np.array(X['Contact Angle'])
cons = np.ones((252, 252))*6.53

x_ax , y_ax = np.meshgrid(x, y)

pred = np.zeros((252, 252))

x_ax = x_ax.reshape(-1)
y_ax = y_ax.reshape(-1)
cons = cons.reshape(-1)
x_ax.shape
data = {'Surface Tension': x_ax, 'Contact Angle': y_ax, 'viscosity': cons}
temp = pd.DataFrame(data)
pred = rf.predict(temp)
z_ax = pred[:,0]
z_ax = z_ax.reshape(252,-1)
x_ax = x_ax.reshape(252,-1)
y_ax = y_ax.reshape(252,-1)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_x = list(unique_points)

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_y = list(unique_points)

X_ax = []  


for i in in_x:
	for j in in_y:
		X_ax.append(x_ax[i][j])

Y_ax = []  

for i in in_x:
	for j in in_y:
		Y_ax.append(y_ax[i][j])

Z_ax = []  

for i in in_x:
	for j in in_y:
		Z_ax.append(z_ax[i][j])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surface = ax.scatter(X_ax, Y_ax, Z_ax, color='red')

ax.set_xlabel('Surface Tension')
ax.set_ylabel('Contact Angle')
ax.set_zlabel('TA')

plt.show()





# ST_CA_AW

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_csv('BTP_4.csv')

features_input = np.array(['Surface Tension','Contact Angle','viscosity'])
features_output = np.array(['TA','AW','TL','FL'])

X = df[features_input]
Y = df[features_output]

bootstrap = [False]
n_estimators = [12]
max_depth = [20]
min_samples_split = [3]
min_samples_leaf = [1]

rf = RandomForestRegressor(bootstrap=False,n_estimators=12,max_depth=20,min_samples_leaf=1,min_samples_split=3)

rf.fit(X,Y)

x = np.array(X['Surface Tension'])
y = np.array(X['Contact Angle'])
cons = np.ones((252, 252))*6.53

x_ax , y_ax = np.meshgrid(x, y)

pred = np.zeros((252, 252))

x_ax = x_ax.reshape(-1)
y_ax = y_ax.reshape(-1)
cons = cons.reshape(-1)
x_ax.shape
data = {'Surface Tension': x_ax, 'Contact Angle': y_ax, 'viscosity': cons}
temp = pd.DataFrame(data)
pred = rf.predict(temp)
z_ax = pred[:,1]
z_ax = z_ax.reshape(252,-1)
x_ax = x_ax.reshape(252,-1)
y_ax = y_ax.reshape(252,-1)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_x = list(unique_points)

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_y = list(unique_points)

X_ax = []  


for i in in_x:
	for j in in_y:
		X_ax.append(x_ax[i][j])

Y_ax = []  

for i in in_x:
	for j in in_y:
		Y_ax.append(y_ax[i][j])

Z_ax = []  

for i in in_x:
	for j in in_y:
		Z_ax.append(z_ax[i][j])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surface = ax.scatter(X_ax, Y_ax, Z_ax, color='red')

ax.set_xlabel('Surface Tension')
ax.set_ylabel('Contact Angle')
ax.set_zlabel('AW')

plt.show()




# ST_CA_FL

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_csv('BTP_4.csv')

features_input = np.array(['Surface Tension','Contact Angle','viscosity'])
features_output = np.array(['TA','AW','TL','FL'])

X = df[features_input]
Y = df[features_output]

bootstrap = [False]
n_estimators = [12]
max_depth = [20]
min_samples_split = [3]
min_samples_leaf = [1]

rf = RandomForestRegressor(bootstrap=False,n_estimators=12,max_depth=20,min_samples_leaf=1,min_samples_split=3)

rf.fit(X,Y)

x = np.array(X['Surface Tension'])
y = np.array(X['Contact Angle'])
cons = np.ones((252, 252))*6.53

x_ax , y_ax = np.meshgrid(x, y)

pred = np.zeros((252, 252))

x_ax = x_ax.reshape(-1)
y_ax = y_ax.reshape(-1)
cons = cons.reshape(-1)
x_ax.shape
data = {'Surface Tension': x_ax, 'Contact Angle': y_ax, 'viscosity': cons}
temp = pd.DataFrame(data)
pred = rf.predict(temp)
z_ax = pred[:,3]
z_ax = z_ax.reshape(252,-1)
x_ax = x_ax.reshape(252,-1)
y_ax = y_ax.reshape(252,-1)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_x = list(unique_points)

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_y = list(unique_points)

X_ax = []  


for i in in_x:
	for j in in_y:
		X_ax.append(x_ax[i][j])

Y_ax = []  

for i in in_x:
	for j in in_y:
		Y_ax.append(y_ax[i][j])

Z_ax = []  

for i in in_x:
	for j in in_y:
		Z_ax.append(z_ax[i][j])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surface = ax.scatter(X_ax, Y_ax, Z_ax, color='red')

ax.set_xlabel('Surface Tension')
ax.set_ylabel('Contact Angle')
ax.set_zlabel('FL')

plt.show()



# # ST_VI_TA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_csv('BTP_4.csv')

features_input = np.array(['Surface Tension','Contact Angle','viscosity'])
features_output = np.array(['TA','AW','TL','FL'])

X = df[features_input]
Y = df[features_output]

bootstrap = [False]
n_estimators = [12]
max_depth = [20]
min_samples_split = [3]
min_samples_leaf = [1]

rf = RandomForestRegressor(bootstrap=False,n_estimators=12,max_depth=20,min_samples_leaf=1,min_samples_split=3)

rf.fit(X,Y)

x = np.array(X['Surface Tension'])
y = np.array(X['viscosity'])
cons = np.ones((252, 252))*25

x_ax , y_ax = np.meshgrid(x, y)

pred = np.zeros((252, 252))

x_ax = x_ax.reshape(-1)
y_ax = y_ax.reshape(-1)
cons = cons.reshape(-1)
x_ax.shape
data = {'Surface Tension': x_ax, 'Contact Angle': cons, 'viscosity': y_ax}
temp = pd.DataFrame(data)
pred = rf.predict(temp)
z_ax = pred[:,0]
z_ax = z_ax.reshape(252,-1)
x_ax = x_ax.reshape(252,-1)
y_ax = y_ax.reshape(252,-1)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_x = list(unique_points)

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_y = list(unique_points)

X_ax = []  


for i in in_x:
	for j in in_y:
		X_ax.append(x_ax[i][j])

Y_ax = []  

for i in in_x:
	for j in in_y:
		Y_ax.append(y_ax[i][j])

Z_ax = []  

for i in in_x:
	for j in in_y:
		Z_ax.append(z_ax[i][j])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surface = ax.scatter(X_ax, Y_ax, Z_ax, color='red')

ax.set_xlabel('Surface Tension')
ax.set_ylabel('viscosity')
ax.set_zlabel('TA')

plt.show()



# # ST_VI_AW

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_csv('BTP_4.csv')

features_input = np.array(['Surface Tension','Contact Angle','viscosity'])
features_output = np.array(['TA','AW','TL','FL'])

X = df[features_input]
Y = df[features_output]

bootstrap = [False]
n_estimators = [12]
max_depth = [20]
min_samples_split = [3]
min_samples_leaf = [1]

rf = RandomForestRegressor(bootstrap=False,n_estimators=12,max_depth=20,min_samples_leaf=1,min_samples_split=3)

rf.fit(X,Y)

x = np.array(X['Surface Tension'])
y = np.array(X['viscosity'])
cons = np.ones((252, 252))*25

x_ax , y_ax = np.meshgrid(x, y)

pred = np.zeros((252, 252))

x_ax = x_ax.reshape(-1)
y_ax = y_ax.reshape(-1)
cons = cons.reshape(-1)
x_ax.shape
data = {'Surface Tension': x_ax, 'Contact Angle': cons, 'viscosity': y_ax}
temp = pd.DataFrame(data)
pred = rf.predict(temp)
z_ax = pred[:,1]
z_ax = z_ax.reshape(252,-1)
x_ax = x_ax.reshape(252,-1)
y_ax = y_ax.reshape(252,-1)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_x = list(unique_points)

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_y = list(unique_points)

X_ax = []  


for i in in_x:
	for j in in_y:
		X_ax.append(x_ax[i][j])

Y_ax = []  

for i in in_x:
	for j in in_y:
		Y_ax.append(y_ax[i][j])

Z_ax = []  

for i in in_x:
	for j in in_y:
		Z_ax.append(z_ax[i][j])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surface = ax.scatter(X_ax, Y_ax, Z_ax, color='red')

ax.set_xlabel('Surface Tension')
ax.set_ylabel('viscosity')
ax.set_zlabel('AW')

plt.show()



# # ST_VI_TL

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_csv('BTP_4.csv')

features_input = np.array(['Surface Tension','Contact Angle','viscosity'])
features_output = np.array(['TA','AW','TL','FL'])

X = df[features_input]
Y = df[features_output]

bootstrap = [False]
n_estimators = [12]
max_depth = [20]
min_samples_split = [3]
min_samples_leaf = [1]

rf = RandomForestRegressor(bootstrap=False,n_estimators=12,max_depth=20,min_samples_leaf=1,min_samples_split=3)

rf.fit(X,Y)

x = np.array(X['Surface Tension'])
y = np.array(X['viscosity'])
cons = np.ones((252, 252))*25

x_ax , y_ax = np.meshgrid(x, y)

pred = np.zeros((252, 252))

x_ax = x_ax.reshape(-1)
y_ax = y_ax.reshape(-1)
cons = cons.reshape(-1)
x_ax.shape
data = {'Surface Tension': x_ax, 'Contact Angle': cons, 'viscosity': y_ax}
temp = pd.DataFrame(data)
pred = rf.predict(temp)
z_ax = pred[:,2]
z_ax = z_ax.reshape(252,-1)
x_ax = x_ax.reshape(252,-1)
y_ax = y_ax.reshape(252,-1)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_x = list(unique_points)

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_y = list(unique_points)

X_ax = []  


for i in in_x:
	for j in in_y:
		X_ax.append(x_ax[i][j])

Y_ax = []  

for i in in_x:
	for j in in_y:
		Y_ax.append(y_ax[i][j])

Z_ax = []  

for i in in_x:
	for j in in_y:
		Z_ax.append(z_ax[i][j])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surface = ax.scatter(X_ax, Y_ax, Z_ax, color='red')

ax.set_xlabel('Surface Tension')
ax.set_ylabel('viscosity')
ax.set_zlabel('TL')

plt.show()



# # ST_VI_FL

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_csv('BTP_4.csv')

features_input = np.array(['Surface Tension','Contact Angle','viscosity'])
features_output = np.array(['TA','AW','TL','FL'])

X = df[features_input]
Y = df[features_output]

bootstrap = [False]
n_estimators = [12]
max_depth = [20]
min_samples_split = [3]
min_samples_leaf = [1]

rf = RandomForestRegressor(bootstrap=False,n_estimators=12,max_depth=20,min_samples_leaf=1,min_samples_split=3)

rf.fit(X,Y)

x = np.array(X['Surface Tension'])
y = np.array(X['viscosity'])
cons = np.ones((252, 252))*25

x_ax , y_ax = np.meshgrid(x, y)

pred = np.zeros((252, 252))

x_ax = x_ax.reshape(-1)
y_ax = y_ax.reshape(-1)
cons = cons.reshape(-1)
x_ax.shape
data = {'Surface Tension': x_ax, 'Contact Angle': cons, 'viscosity': y_ax}
temp = pd.DataFrame(data)
pred = rf.predict(temp)
z_ax = pred[:,3]
z_ax = z_ax.reshape(252,-1)
x_ax = x_ax.reshape(252,-1)
y_ax = y_ax.reshape(252,-1)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_x = list(unique_points)

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_y = list(unique_points)

X_ax = []  


for i in in_x:
	for j in in_y:
		X_ax.append(x_ax[i][j])

Y_ax = []  

for i in in_x:
	for j in in_y:
		Y_ax.append(y_ax[i][j])

Z_ax = []  

for i in in_x:
	for j in in_y:
		Z_ax.append(z_ax[i][j])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surface = ax.scatter(X_ax, Y_ax, Z_ax, color='red')

ax.set_xlabel('Surface Tension')
ax.set_ylabel('viscosity')
ax.set_zlabel('FL')

plt.show()



# # CA_VI_TA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_csv('BTP_4.csv')

features_input = np.array(['Surface Tension','Contact Angle','viscosity'])
features_output = np.array(['TA','AW','TL','FL'])

X = df[features_input]
Y = df[features_output]

bootstrap = [False]
n_estimators = [12]
max_depth = [20]
min_samples_split = [3]
min_samples_leaf = [1]

rf = RandomForestRegressor(bootstrap=False,n_estimators=12,max_depth=20,min_samples_leaf=1,min_samples_split=3)

rf.fit(X,Y)

x = np.array(X['Contact Angle'])
y = np.array(X['viscosity'])
cons = np.ones((252, 252))*27.24

x_ax , y_ax = np.meshgrid(x, y)

pred = np.zeros((252, 252))

x_ax = x_ax.reshape(-1)
y_ax = y_ax.reshape(-1)
cons = cons.reshape(-1)
x_ax.shape
data = {'Surface Tension': cons, 'Contact Angle': x_ax, 'viscosity': y_ax}
temp = pd.DataFrame(data)
pred = rf.predict(temp)
z_ax = pred[:,0]
z_ax = z_ax.reshape(252,-1)
x_ax = x_ax.reshape(252,-1)
y_ax = y_ax.reshape(252,-1)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_x = list(unique_points)

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_y = list(unique_points)

X_ax = []  


for i in in_x:
	for j in in_y:
		X_ax.append(x_ax[i][j])

Y_ax = []  

for i in in_x:
	for j in in_y:
		Y_ax.append(y_ax[i][j])

Z_ax = []  

for i in in_x:
	for j in in_y:
		Z_ax.append(z_ax[i][j])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surface = ax.scatter(X_ax, Y_ax, Z_ax, color='red')

ax.set_xlabel('Contact Angle')
ax.set_ylabel('viscosity')
ax.set_zlabel('TA')

plt.show()




# # CA_VI_AW

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_csv('BTP_4.csv')

features_input = np.array(['Surface Tension','Contact Angle','viscosity'])
features_output = np.array(['TA','AW','TL','FL'])

X = df[features_input]
Y = df[features_output]

bootstrap = [False]
n_estimators = [12]
max_depth = [20]
min_samples_split = [3]
min_samples_leaf = [1]

rf = RandomForestRegressor(bootstrap=False,n_estimators=12,max_depth=20,min_samples_leaf=1,min_samples_split=3)

rf.fit(X,Y)

x = np.array(X['Contact Angle'])
y = np.array(X['viscosity'])
cons = np.ones((252, 252))*27.24

x_ax , y_ax = np.meshgrid(x, y)

pred = np.zeros((252, 252))

x_ax = x_ax.reshape(-1)
y_ax = y_ax.reshape(-1)
cons = cons.reshape(-1)
x_ax.shape
data = {'Surface Tension': cons, 'Contact Angle': x_ax, 'viscosity': y_ax}
temp = pd.DataFrame(data)
pred = rf.predict(temp)
z_ax = pred[:,1]
z_ax = z_ax.reshape(252,-1)
x_ax = x_ax.reshape(252,-1)
y_ax = y_ax.reshape(252,-1)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_x = list(unique_points)

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_y = list(unique_points)

X_ax = []  


for i in in_x:
	for j in in_y:
		X_ax.append(x_ax[i][j])

Y_ax = []  

for i in in_x:
	for j in in_y:
		Y_ax.append(y_ax[i][j])

Z_ax = []  

for i in in_x:
	for j in in_y:
		Z_ax.append(z_ax[i][j])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surface = ax.scatter(X_ax, Y_ax, Z_ax, color='red')

ax.set_xlabel('Contact Angle')
ax.set_ylabel('viscosity')
ax.set_zlabel('AW')

plt.show()




# # CA_VI_TL

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_csv('BTP_4.csv')

features_input = np.array(['Surface Tension','Contact Angle','viscosity'])
features_output = np.array(['TA','AW','TL','FL'])

X = df[features_input]
Y = df[features_output]

bootstrap = [False]
n_estimators = [12]
max_depth = [20]
min_samples_split = [3]
min_samples_leaf = [1]

rf = RandomForestRegressor(bootstrap=False,n_estimators=12,max_depth=20,min_samples_leaf=1,min_samples_split=3)

rf.fit(X,Y)

x = np.array(X['Contact Angle'])
y = np.array(X['viscosity'])
cons = np.ones((252, 252))*27.24

x_ax , y_ax = np.meshgrid(x, y)

pred = np.zeros((252, 252))

x_ax = x_ax.reshape(-1)
y_ax = y_ax.reshape(-1)
cons = cons.reshape(-1)
x_ax.shape
data = {'Surface Tension': cons, 'Contact Angle': x_ax, 'viscosity': y_ax}
temp = pd.DataFrame(data)
pred = rf.predict(temp)
z_ax = pred[:,2]
z_ax = z_ax.reshape(252,-1)
x_ax = x_ax.reshape(252,-1)
y_ax = y_ax.reshape(252,-1)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_x = list(unique_points)

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_y = list(unique_points)

X_ax = []  


for i in in_x:
	for j in in_y:
		X_ax.append(x_ax[i][j])

Y_ax = []  

for i in in_x:
	for j in in_y:
		Y_ax.append(y_ax[i][j])

Z_ax = []  

for i in in_x:
	for j in in_y:
		Z_ax.append(z_ax[i][j])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surface = ax.scatter(X_ax, Y_ax, Z_ax, color='red')

ax.set_xlabel('Contact Angle')
ax.set_ylabel('viscosity')
ax.set_zlabel('TL')

plt.show()




# CA_VI_FL

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

df = pd.read_csv('BTP_4.csv')

features_input = np.array(['Surface Tension','Contact Angle','viscosity'])
features_output = np.array(['TA','AW','TL','FL'])

X = df[features_input]
Y = df[features_output]

bootstrap = [False]
n_estimators = [12]
max_depth = [20]
min_samples_split = [3]
min_samples_leaf = [1]

rf = RandomForestRegressor(bootstrap=False,n_estimators=12,max_depth=20,min_samples_leaf=1,min_samples_split=3)

rf.fit(X,Y)

x = np.array(X['Contact Angle'])
y = np.array(X['viscosity'])
cons = np.ones((252, 252))*27.24

x_ax , y_ax = np.meshgrid(x, y)

pred = np.zeros((252, 252))

x_ax = x_ax.reshape(-1)
y_ax = y_ax.reshape(-1)
cons = cons.reshape(-1)
x_ax.shape
data = {'Surface Tension': cons, 'Contact Angle': x_ax, 'viscosity': y_ax}
temp = pd.DataFrame(data)
pred = rf.predict(temp)
z_ax = pred[:,3]
z_ax = z_ax.reshape(252,-1)
x_ax = x_ax.reshape(252,-1)
y_ax = y_ax.reshape(252,-1)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_x = list(unique_points)

unique_points = set()

while len(unique_points) < 100:
    unique_points.add(random.randint(0, 240))

in_y = list(unique_points)

X_ax = []  


for i in in_x:
	for j in in_y:
		X_ax.append(x_ax[i][j])

Y_ax = []  

for i in in_x:
	for j in in_y:
		Y_ax.append(y_ax[i][j])

Z_ax = []  

for i in in_x:
	for j in in_y:
		Z_ax.append(z_ax[i][j])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

surface = ax.scatter(X_ax, Y_ax, Z_ax, color='red')

ax.set_xlabel('Contact Angle')
ax.set_ylabel('viscosity')
ax.set_zlabel('FL')

plt.show()
