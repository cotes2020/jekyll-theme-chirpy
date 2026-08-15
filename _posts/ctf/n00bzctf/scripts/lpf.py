def largest_prime_factor(n): 
	""" 
	Find the largest prime factor of a positive integer 'n' 
	:param n: positive integer ( 1 <= n <= 10^15) 
	:return: largest prime factor of n 
	"""
	largest_prime = -1
	i = 2
	while i * i <= n: 
		while n % i == 0: 
			largest_prime = i 
			n = n // i 
		i = i + 1
	if n > 1: 
		largest_prime = n 
	return largest_prime 


