import numpy as np

print("Questions 1:")
print("Simple Return: 5%")

log_return = np.log(105/100)

print("log return is: ", log_return)




print("Questions 2:")
print("Simple Return: 10%")



log_return = np.log(55/50)
print("log return is: ", log_return)

print("Questions 3:")


def return_simple_calculator(initial_price, rate, days):
    return initial_price * (1 + rate)**days


print("Simple Return: ", return_simple_calculator(200, 0.02, 5))

print("Simple Return: ", return_simple_calculator(200, 0.02, 10))



def return_log_calculator(initial_price, rate, days):
    return initial_price * np.exp(rate * days)


print("log return is: ", return_log_calculator(150, 0.01, 10))
print("log return is: ", return_log_calculator(150, 0.01, 20))





print("Questions 4:")
simple_rate= 1.0005**5
print("simple rate is: ", simple_rate)


log_rate= np.log(1.0005)
print("log rate is: ", log_rate)

log_additive = 0.02-0.01+0.015
finalvalue = 120 * np.exp(log_additive)
print("final value is: ", finalvalue)

simple_return = (finalvalue/120)-1
print("simple return is: ", simple_return)


print("Questions 5:")
montly_log = np.log(1.08)
print("montly log is: ", montly_log)


r = 0.08/20
print("daily r is ", r)


print("daily log is: ", np.exp(0.005 * 252))


print("Questions 6:")






