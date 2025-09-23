class dhcp3_fmri:
    def __init__(self, data):
        self.data = data

    def get_fmri(self):
        return self.data.get('fmri', None)

    def set_fmri(self, fmri):
        self.data['fmri'] = fmri