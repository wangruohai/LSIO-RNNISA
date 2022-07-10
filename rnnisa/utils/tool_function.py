"""
This module contains some tool function

Author:
    Tan Wang
"""

from pickle import dump, load
from time import time
from email.mime.text import MIMEText
import smtplib



def my_dump(obj, path):
    with open(path, 'wb') as f:
        dump(obj, f)
        f.close()


def my_load(path):
    with open(path, 'rb') as f:
        obj = load(f, encoding='bytes')
        f.close()
    return obj


def print_run_time(name, t_start, time_unit=None, num=1.0):
    run_time = (time() - t_start) / num
    if time_unit is None:
        if run_time < 60:
            print('Run Time for', name, ': %6.2f seconds' % run_time)
        elif run_time < 3600:
            print('Run Time for', name, ': %6.2f minutes' % (run_time / 60))
        else:
            print('Run Time for', name, ': %6.2f hours' % (run_time / 3600))
    elif time_unit == 'second':
        print('Run Time for', name, ': %6.2f seconds' % run_time)
    elif time_unit == 'minute':
        print('Run Time for', name, ': %6.2f minutes' % (run_time / 60))
    elif time_unit == 'hour':
        print('Run Time for', name, ': %6.2f hours' % (run_time / 3600))
    else:
        print('Invalid Time Unit!')


def send_email():
    msg = MIMEText('程序已经执行完毕，请及时上线查看', 'plain', 'utf-8')
    msg['From'] = 'wt_smtp@163.com'
    msg['To'] = 'wt_smtp@163.com'
    msg['Subject'] = '您的程序已经执行完毕'
    from_addr = 'wt_smtp@163.com'
    passward = 'UYJJWIZVDBFZDYZE'
    to_addr = 'wt_smtp@163.com'
    smtp_server = 'smtp.163.com'
    server = smtplib.SMTP_SSL(smtp_server)
    server.set_debuglevel(1)
    server.login(from_addr, passward)
    server.sendmail(from_addr, [to_addr], msg.as_string())
    server.quit()