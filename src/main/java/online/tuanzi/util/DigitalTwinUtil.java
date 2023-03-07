package online.tuanzi.util;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;

import java.util.ArrayList;
import java.util.List;

public class DigitalTwinUtil {
    private List<Double> x0;//原始数据列x0
    private List<Double> x1;//累加序列x1
    private RealMatrix matrix_B;//数据矩阵B
    private RealMatrix matrix_y;//矩阵y
    private RealMatrix fittingVector_U;//拟合向量U

    public void setX0(List<Double> x0) {
        this.x0 = x0;
    }

    public List<Double> getX0() {
        return x0;
    }

    //1.由原始数据列计算一次累加序列x1并输出结果
    //获取原始数据列x0
    private List<Double> getInitData(){
        x0 = List.of(2.874, 3.278, 3.337, 3.390, 3.679);
//        this.x0 = x0;
        return x0;
    }

    //获取累加序列x1
    private List<Double> getCumulativeSequence(){
        List<Double> cumulativeSequence = new ArrayList<>();

        //x00 == x10
        cumulativeSequence.add(x0.get(0));

        for (int i = 1; i < x0.size(); i++) {
            //x1i == x1(i-1) + x0i (i > 1)
            cumulativeSequence.add(i, cumulativeSequence.get(i - 1) + x0.get(i));
        }
        x1 = cumulativeSequence;

        return x1;
    }

    //
    public void printResult(){
//        System.out.println("原始数据列x0："+getInitData());
        System.out.println("原始数据列x0："+getX0());

        System.out.println("累加序列x1："+getCumulativeSequence());

        System.out.println("数据矩阵B:"+buildMatrix_B());

        System.out.println("数据矩阵y:"+buildMatrix_y());

        System.out.println("拟合向量U:"+calculateFittingVector_U());

        System.out.println("x1的拟合值："+calculateForecast_x1());

        System.out.println("x0的拟合值："+calculateForecast_x0());

        System.out.println("残差："+residual());

        System.out.println("相对残差："+relativeResidual());

        System.out.println("x0的均值："+mean_x0());

        System.out.println("x0的方差："+variance_x0());

        System.out.println("残差的均值："+mean_residual());

        System.out.println("残差的方差："+variance_residual());

        System.out.println("后验差比值："+posteriorDifferenceRatio());
    }

    //2.建立矩阵
    //构建矩阵B
    private RealMatrix buildMatrix_B(){

        //构建n-1行2列的数据矩阵B
        matrix_B = MatrixUtils.createRealMatrix(x1.size() - 1, 2);

        //0 - n-2 ---> 1 - n-1
        for (int i = 0; i < x1.size() - 1; i++) {
            // 第一列：-1/2(x1(i+1)+x1i) 第二列：1
            matrix_B.setEntry(i,0,-0.5*(x1.get(i+1)+x1.get(i)));
            matrix_B.setEntry(i,1,1);
        }
        return matrix_B;
    }

    //构建矩阵y
    private RealMatrix buildMatrix_y(){
        //构建1行n-1列的数据矩阵Y
        RealMatrix matrix_y_transpose = new Array2DRowRealMatrix(1, x0.size() - 1);

        // 0 - n-2 --> 2 - n
        for (int i = 0; i < x0.size() - 1; i++) {
            //第1行第i+1列
            matrix_y_transpose.setEntry(0, i, x0.get(i+1));
        }

        //需要转置
        matrix_y = matrix_y_transpose.transpose();
        return matrix_y;
    }

    //计算拟合向量U
    private RealMatrix calculateFittingVector_U(){

        //计算(BT B)-1：B转置×B之后的逆矩阵
        RealMatrix temp = MatrixUtils.inverse(matrix_B.transpose().multiply(matrix_B));

        //计算拟合向量U
        fittingVector_U = temp.multiply(matrix_B.transpose()).multiply(matrix_y);

        return fittingVector_U;
    }

    /**
     * 计算时间响应方程x1(k+1)
     * @param k ：当k为1-n-1时，x1(k+1)为拟合值，当k>=n时，x1(k+1)为预报值
     * @param a : 估计值a
     * @param b : 估计值b
     * @return
     */
    private Double calculateTimeResponseEquation(int k, double a, double b){

        // (x11-b/a)*e^(-ak) + b/a
//        System.out.printf("时间响应方程为：%fe^(%fk)-%f",(x1.get(0) - b/a),(-a*k), b/a);
        return (x1.get(0) - b/a)*Math.exp(-a*k) + b/a;
    }

    //计算预报值x1(k+1)
    private List<Double> calculateForecast_x1(){

        double a = fittingVector_U.getEntry(0, 0);//第1行第1列
        double b = fittingVector_U.getEntry(1, 0);//第2行第1列
        //x1的拟合值集合
        List<Double> forecast_x1 = new ArrayList<>();
        // 0 - n-1 -->  1 - n
        for (int i = 0; i < x1.size(); i++) {
            forecast_x1.add(calculateTimeResponseEquation(i , a, b));
        }

        return forecast_x1;
    }

    //计算预报值x0(k+1)
    private List<Double> calculateForecast_x0(){
        List<Double> forecast_x1 = calculateForecast_x1();

        //x0的拟合值集合
        List<Double> forecast_x0 = new ArrayList<>();

        // 1 - n
        // 后减前
        for (int i = 1; i < forecast_x1.size(); i++) {
            forecast_x0.add(forecast_x1.get(i) - forecast_x1.get(i-1));
        }

        return forecast_x0;
    }

    //3.精度检验
    //3.1 残差检验
    private List<Double> residual(){
        List<Double> forecast_x0 = calculateForecast_x0();//2-n
        List<Double> residual = new ArrayList<>();

        for (int i = 0; i < forecast_x0.size(); i++) {
            residual.add(x0.get(i+1) - forecast_x0.get(i));
        }
        return residual;
    }

    //残差计算Relative
    private List<Double> relativeResidual(){
        List<Double> residual = residual();
        List<Double> relativeResidual = new ArrayList<>();
        for (int i = 0; i < residual.size(); i++) {
            relativeResidual.add(residual.get(i) / x0.get(i+1));
        }
        return relativeResidual;
    }

    //相对残差
    //3.2 后验差检验
    //x0的均值mean X-
    private Double mean_x0(){
        return StatUtils.mean(x0.stream().mapToDouble(Double::doubleValue).toArray());
    }

    //x0的方差variance S1
    private Double variance_x0(){
//        return StatUtils.populationVariance(x0.stream().mapToDouble(Double::doubleValue).toArray(), mean_x0(), 0,x0.size());
        return new StandardDeviation(false).evaluate(x0.stream().mapToDouble(Double::doubleValue).toArray());
    }

    //残差的均值 E-
    private Double mean_residual(){
        List<Double> residual = residual();

        return StatUtils.mean(residual.stream().mapToDouble(Double::doubleValue).toArray());
    }

    //残差的方差 S2
    private Double variance_residual(){
        List<Double> residual = residual();

//        return StatUtils.variance(residual.stream().mapToDouble(Double::doubleValue).toArray(), mean_residual());
        return new StandardDeviation(false).evaluate(residual.stream().mapToDouble(Double::doubleValue).toArray(), mean_residual());
    }

    //后验差比值
    private Double posteriorDifferenceRatio(){
        Double S1 = variance_x0();
        Double S2 = variance_residual();

        Double C = S2 / S1;

        return C;
    }
}
