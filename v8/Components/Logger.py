class Logger:
    def __init__(self, filename, log_level):
        self.logs_folder = "logs/"
        self.filename = filename
        self.classifications = {
            4: "[INFO]: ",             #level 4
            3: "[WARNING]: ",          #level 3
            2: "[ERROR]: ",            #level 2
            1: "[DEBUG]: "             #level 1
        }
        self.log_level = log_level

    def log(self, message, classification, Loud):
        # Check if folder exists, if not create it
        import os
        if not os.path.exists(self.logs_folder):
            os.makedirs(self.logs_folder)

        try:
            file = open(self.logs_folder + self.filename, "a")
        except FileNotFoundError:
            #create the file if it does not exist
            try:
                file = open(self.logs_folder + self.filename, "w")
            except Exception as e:
                print(f"Failed to create log file: {e}")
                return

        except Exception as e:
            print(f"Failed to open log file: {e}")
            return
        if classification not in self.classifications:
            raise ValueError("Invalid classification level.")
        prefix = self.classifications[classification]
        formatted_message = f"{prefix}{message}\n"
        
        if classification <= self.log_level:
            file.write(formatted_message)
        
        if Loud:
            print(formatted_message, end='')
        file.close()

