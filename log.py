from datetime import datetime,date

class Log:
    def logger(self,msg):
        try:
            time = datetime.now().strftime("%H:%M:%S")
            dateToday = date.today()
            with open("log.txt","a+") as writer:
                writer.write(f"{dateToday} {time}${msg}\n")
        except Exception as e:
            raise e
        