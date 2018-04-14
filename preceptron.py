class Preceptron(object):
	def __init__(self, input_num, activation):
		self.activation = activation
		self.weights = [0.0 for _ in range(input_num)]
		self.bias = 0;

	def __str__(self):
		return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)


	# Calculate the prediction
	def predict(self, input_vector):
		# wrap input and weights like (x_i, w_i)
		# use map to get calculate x_i * w_i
		# use reduce to get the sum
		return self.activation(reduce(lambda a, b: a+b, map(lambda x, w: x*w, zip(input_vector, self.weights)),0.0)+self.bias)

	def _one_iteration(self, input_vector, targets, learning_rate)
		samples = zip(input_vector, targets)
		for(input_vector, targets) in samples:
			predictions = self.predict(input_vector)
			# Update weights and bias
			self._update_weight(input_vector, predictions, targets, learning_rate)

	def _update_weights(self, input_vector, predictions, targets, learning_rate):
		delta = target - predictions
		self.weights = map(lambda x, w: w + rate * delta * x, zip(input_vector, self,weights))
		self.bias += learning_rate * delta

	