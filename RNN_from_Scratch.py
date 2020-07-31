import numpy as np
def tanh(x):
	return np.tanh(x)

def softmax(x):
	return np.exp(x)/np.sum(np.exp(x))

# Single RNN cell
def rnn(a_prev, x_t, parameters):

	a_next = tanh(parameters['Wax'].dot(x_t) + parameters['Waa'].dot(a_prev) + parameters['ba'])

	y_t = softmax(parameters['Wya'].dot(a_next) + parameters['by'])

	info = (a_prev, x_t, a_next, parameters)
	return a_next, y_t, info


# forward pass through t_x timesteps
def forward(X, parameters):
	nx, m, t_x = X.shape
	ny, na = parameters['Wya']

	A = np.zeros((na, m, t_x))
	Y = np.zeros((ny, m, t_x))
	infos = []
	a_prev = A[:, :, 0]

	for t in range(t_x):
		x_t = X[:, :, t]
		a_next, y_t, info = rnn(a_prev, x_t, parameters)

		A[:, :, t] = a_next
		Y[:, :, t] = y_t
		a_prev = a_next
		infos.append(info)


	return A, Y, (infos, X)