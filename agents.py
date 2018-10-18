# for the dynamic operation of the network, that is, when performing the
# time modeling

class WPP(object):
    def __init__(self, power_plant: str, location: tuple, operating_capacity: float, fuel_type='Wind',
                 technology_type='Wind Turbine') -> None:

        self.power_plant = str(power_plant)
        self.technology_type = str(technology_type)
        self.latitude, self.longitude = location
        self.operating_capacity = operating_capacity
        self.fuel_type = fuel_type

    def __str__(self):
        return f'{self.power_plant}: Type: {self.technology_type}, operating capacity: {self.operating_capacity},' \
               f' location: {self.latitude, self.longitude}'


class NGPP(object):
    def __init__(self, power_plant: str, location: tuple, operating_capacity: float, technology_type: str,
                 fuel_type='Natural Gas'):

        self.power_plant = power_plant
        self.technology_type = technology_type
        self.latitude, self.longitude = location
        self.operating_capacity = operating_capacity
        self.fuel_type = fuel_type

    def __str__(self):
        return f'{self.power_plant}: Type: {self.technology_type}, operating capacity: {self.operating_capacity},' \
               f' location: {self.latitude, self.longitude}'


class Bilateral_Contract(object):
    def __init__(self):
        pass


class Pairing(object):
    pass

