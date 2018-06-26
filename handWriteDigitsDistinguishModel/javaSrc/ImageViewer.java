import java.awt.*;
import java.io.*;
import javax.swing.*;

public class ImageViewer {
    public static void main(String[] args) {
        EventQueue.invokeLater(() -> {
            JFrame frame = new ImageViewerFrame();
            frame.setTitle("手写识别系统");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setVisible(true);
        });
    }
}

class ImageViewerFrame extends JFrame {
    private JLabel label;
    private JFileChooser chooser;
    private static final int DEFAULT_WIDTH = 600;
    private static final int DEFAULT_HEIGHT = 768;

	private void welcomePage() {
		ImageIcon icon = new ImageIcon("./welcomePage.jpg");
        icon.setImage(icon.getImage().getScaledInstance(DEFAULT_WIDTH, DEFAULT_HEIGHT, Image.SCALE_DEFAULT));
        label.setIcon(icon);
	}
	
    private void initOptionPane() {
        UIManager.put("Label.font", new Font("Dialog", Font.PLAIN,78));
    }

    private String getAnswer(String path) {
        PyCaller.writeImagePath(path);
        PyCaller.execPy();
        return PyCaller.readAnswer();
    }

    public ImageViewerFrame() {
        setSize(DEFAULT_WIDTH, DEFAULT_HEIGHT);
        setLocationRelativeTo(null);

        // use a label to display the images
        label = new JLabel();
        add(label);

        // set up the file chooser
        chooser = new JFileChooser();
        chooser.setCurrentDirectory(new File("."));

        // set up the menu bar
        JMenuBar menuBar = new JMenuBar();
        setJMenuBar(menuBar);

        JMenu menu = new JMenu("File");
        menuBar.add(menu);

        JMenuItem openItem = new JMenuItem("Open");
        menu.add(openItem);
		
		welcomePage();
		
        openItem.addActionListener(event -> {
            // show file chooser dialog
            int result = chooser.showOpenDialog(null);

            // if file selected, set it as icon of the label
            if (result == JFileChooser.APPROVE_OPTION)
            {
                String path = chooser.getSelectedFile().getPath();
                ImageIcon icon = new ImageIcon(path);
                icon.setImage(icon.getImage().getScaledInstance(DEFAULT_WIDTH, DEFAULT_HEIGHT, Image.SCALE_DEFAULT));
                label.setIcon(icon);

                // my codes
                initOptionPane();
                String answer = getAnswer(path);
                JOptionPane.showMessageDialog(null, "识别结果：" + answer + " ");
            }
        });

        JMenuItem exitItem = new JMenuItem("Exit");
        menu.add(exitItem);
        exitItem.addActionListener(event -> System.exit(0));
    }
}
