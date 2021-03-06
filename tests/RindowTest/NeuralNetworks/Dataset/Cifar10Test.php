<?php
namespace RindowTest\NeuralNetworks\Dataset\Cifar10Test;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use SplFixedArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;

/**
 * @requires extension rindow_openblas
 */
class Test extends TestCase
{
    protected $plot = false;

    public function setUp() : void
    {
        $this->plot = true;
        $this->pickleFile = sys_get_temp_dir().'/rindow/nn/datasets/cifar-10-batches-bin/cifar10.pkl';
    }

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('TRAVIS_PHP_VERSION') ? true : false,
        ];
    }

    public function testDownloadFiles()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $nn->datasets()->cifar10()->downloadFiles();
        $this->assertTrue(true);
    }

    public function testLoadDataFromFiles()
    {
        $pickleFile = $this->pickleFile;
        if(file_exists($pickleFile)) {
            unlink($pickleFile);
            sleep(1);
        }

        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        $plt = new Plot($this->getPlotConfig(),$mo);

        [[$train_img,$train_label],[$test_img,$test_label]] =
            $nn->datasets()->cifar10()->loadData();

        sleep(1);
        $this->assertTrue(file_exists($pickleFile));

        if($this->plot) {
            [$figure, $axes] = $plt->subplots(5,7);
            for($i=0;$i<count($axes);$i++) {
                $axes[$i]->setAspect('equal');
                $axes[$i]->setFrame(false);
                $axes[$i]->imshow($train_img[$i],
                    null,null,null,$origin='upper');
            }
            $plt->show();
        }
    }

    public function testLoadDataFromPickle()
    {
        $pickleFile = $this->pickleFile;
        $this->assertTrue(file_exists($pickleFile));

        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        //$config = [
        //    'figure.bgColor' => 'white',
        //    'figure.figsize' => [500,500],
        //    'figure.leftMargin' => 0,
        //    'figure.bottomMargin' => 0,
        //    'figure.rightMargin' => 0,
        //    'figure.topMargin' => 0,
        //];
        $plt = new Plot($this->getPlotConfig(),$mo);

        [[$train_img,$train_label],[$test_img,$test_label]] =
            $nn->datasets()->cifar10()->loadData();

        if($this->plot) {
            [$figure, $axes] = $plt->subplots(5,7);
            for($i=0;$i<count($axes);$i++) {
                $axes[$i]->setAspect('equal');
                $axes[$i]->setFrame(false);
                $axes[$i]->imshow($train_img[$i],null,null,null,$origin='upper');
            }
            $plt->show();
        }
    }

    public function testCleanPickle()
    {
        $pickleFile = $this->pickleFile;
        $this->assertTrue(file_exists($pickleFile));

        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);
        $nn->datasets()->cifar10()->cleanPickle();
        $this->assertTrue(true);
    }
}
