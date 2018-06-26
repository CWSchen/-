import java.io.*;

class PyCaller {
    private static final String DATA_SWAP = "temp.txt";
    private static final String PY_URL = System.getProperty("user.dir") + "\\test.py";

    public static void writeImagePath(String path) {
        PrintWriter pw = null;
        try {
            pw = new PrintWriter(new FileWriter(new File(DATA_SWAP)));
        } catch (IOException e) {
            e.printStackTrace();
        }

        pw.print(path);
        pw.close();
    }

    public static String readAnswer() {
        BufferedReader br;
        String answer = null;
        try {
            br = new BufferedReader(new FileReader(new File(DATA_SWAP)));
            answer = br.readLine();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return answer;
    }

    public static void execPy() {
        Process proc = null;
        try {
            System.out.println(System.getProperty("user.dir") + "\\test.py");
            proc = Runtime.getRuntime().exec("ipython3 " + PY_URL);
            proc.waitFor();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        writeImagePath("D:\\labs\\Yao\\test.jpg");
        execPy();
        System.out.println(readAnswer());
    }
}