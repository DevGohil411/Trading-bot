import MetaTrader5 as mt5

print('initialize:', mt5.initialize())
print('init_err:', mt5.last_error())
print('login:', mt5.login(login=10007121964, password='Xs-vWfE0', server='MetaQuotes-Demo'))
print('login_err:', mt5.last_error())
mt5.shutdown()
