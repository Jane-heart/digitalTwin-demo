package online.tuanzi;

import online.tuanzi.util.DigitalTwinUtil;

import java.util.List;

public class DigitalTwinMain {
    public static void main(String[] args) {
        DigitalTwinUtil digitalTwinUtil = new DigitalTwinUtil();
        digitalTwinUtil.setX0(List.of(2.874, 3.278, 3.337, 3.390, 3.679));
        digitalTwinUtil.printResult();
    }
}
