from abc import ABC, abstractmethod

# Base class for LED
class LED(ABC):
    @abstractmethod
    def turn_on(self):
        """Turn the LED on."""
        pass

    @abstractmethod
    def turn_off(self):
        """Turn the LED off."""
        pass

    @abstractmethod
    def set_color(self, color):
        """Set the color of the LED."""
        pass

    @abstractmethod
    def set_brightness(self, brightness):
        """Set the brightness of the LED."""
        pass


# Example vendor-specific LED Class
class VendorALed(LED):
    def __init__(self):
        self.state = False
        self.color = (255, 255, 255)  # Default to white
        self.brightness = 100  # Default brightness

    def turn_on(self):
        self.state = True
        print("Vendor A LED is ON")

    def turn_off(self):
        self.state = False
        print("Vendor A LED is OFF")

    def set_color(self, color):
        self.color = color
        print(f"Vendor A LED color set to {color}")

    def set_brightness(self, brightness):
        self.brightness = brightness
        print(f"Vendor A LED brightness set to {brightness}")


# Room ThingModule controlling multiple LEDs
class RoomThingModule:
    def __init__(self):
        self.leds = []  # List to hold LED instances

    def add_led(self, led: LED):
        """Add an LED instance to the module."""
        self.leds.append(led)

    def turn_on_all(self):
        """Turn on all LEDs in the module."""
        for led in self.leds:
            led.turn_on()

    def turn_off_all(self):
        """Turn off all LEDs in the module."""
        for led in self.leds:
            led.turn_off()

    def set_color_all(self, color):
        """Set the color for all LEDs in the module."""
        for led in self.leds:
            led.set_color(color)

    def set_brightness_all(self, brightness):
        """Set the brightness for all LEDs in the module."""
        for led in self.leds:
            led.set_brightness(brightness)


# Example Usage
if __name__ == "__main__":
    room = RoomThingModule()  # Create a RoomThingModule instance
    
    led1 = VendorALed()  # Create an instance of VendorALed
    room.add_led(led1)  # Add the LED to the room module

    room.turn_on_all()  # Turn on all LEDs
    room.set_color_all((255, 0, 0))  # Set color to red
    room.set_brightness_all(75)  # Set brightness to 75%