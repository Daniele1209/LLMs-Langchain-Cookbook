class ResponseHandler():

    def __init__(self, response=''):
        self.response = response

    def set_response(self, new_reponse):
        self.response = new_reponse
    
    def get_reponse(self):
        return self.response
    
    def post_process_response(self):
        if self.response:
            list_elements = list(filter(str.strip, self.response.split('\n')))
            list_elements = list(filter(lambda x: (x[0].isnumeric() or x[0] == '-') and x[-1] == '.', list_elements))
            print(list_elements)
            self.response = '\n'.join(list_elements)
            return True
        return False