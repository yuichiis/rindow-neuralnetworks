<?php
namespace Rindow\NeuralNetworks\Dataset;

use LogicException;
use Rindow\Math\Matrix\MatrixOperator;
use Interop\Polite\Math\Matrix\NDArray;
use PharData;

class Cifar10
{
    protected $mo;
    protected $baseUrl = 'http://www.cs.toronto.edu/~kriz/';
    protected $downloadFile = 'cifar-10-binary.tar.gz';
    protected $keyFiles = [
        'data_file_1'=>'data_batch_1.bin',
        'data_file_2'=>'data_batch_2.bin',
        'data_file_3'=>'data_batch_3.bin',
        'data_file_4'=>'data_batch_4.bin',
        'data_file_5'=>'data_batch_5.bin',
        'test_file'=>'test_batch.bin',
    ];
    protected $trainNum = 60000;
    protected $testNum = 10000;
    protected $imageShape = [3, 32, 32]; // = 784
    protected $datasetDir;
    protected $saveFile;

    public function __construct($mo)
    {
        $this->matrixOperator = $mo;
        $this->datasetDir = $this->getDatasetDir();
        if(!file_exists($this->datasetDir)) {
            @mkdir($this->datasetDir,0777,true);
        }
        $this->saveFile = $this->datasetDir . "/cifar10.pkl";
    }
    protected function getDatasetDir()
    {
        return sys_get_temp_dir().'/rindow/nn/datasets/cifar-10-batches-bin';
    }

    protected function console($message)
    {
        fwrite(STDERR,$message);
    }

    public function loadData(string $filePath=null)
    {
        $mo = $this->matrixOperator;
        if($filePath===null) {
            $filePath = $this->saveFile;
        }
        if(file_exists($filePath)) {
            $dataset = $this->loadPickle($filePath);
        } else {
            $dataset = $this->getFiles($filePath);
        }

        return [[$dataset['train_images'], $dataset['train_labels']],
                [$dataset['test_images'],  $dataset['test_labels']]];
    }

    public function cleanPickle(string $filePath=null)
    {
        if($filePath===null) {
            $filePath = $this->saveFile;
        }
        unlink($this->saveFile);
    }

    protected function loadPickle($filePath)
    {
        $this->console("Loading pickle file ...");
        $data = file_get_contents($filePath);
        if(!$data)
            throw new LogicException('read error: '.$this->saveFile);
        $dataset = unserialize($data);
        unset($data);
        $this->console("Done!\n");
        return $dataset;
    }

    protected function getFiles($filePath)
    {
        $this->downloadFiles();
        $dataset = $this->convertNDArray();
        $this->console("Creating pickle file ...");
        //with open(save_file, 'wb') as f:
        //    pickle.dump(dataset, f, -1)
        file_put_contents($filePath,serialize($dataset));
        $this->console("Done!\n");
        return $dataset;
    }

    public function downloadFiles()
    {
        $this->download($this->downloadFile);
    }

    protected function download($filename)
    {
        $filePath = $this->datasetDir . "/" . $filename;

        if(file_exists($filePath))
            return;

        $this->console("Downloading " . $filename . " ... ");
        copy($this->baseUrl.$filename, $filePath);
        $this->console("Done\n");
        $this->console("Extract\n");
        $phar = new PharData($filePath);
        $phar->extractTo($this->datasetDir.'/..');
        $this->console("Done\n");
    }

    protected function convertNDArray()
    {
        $mo = $this->matrixOperator;
        $filenames = $this->keyFiles;
        $testFiles = [array_pop($filenames)];
        $dataset['train_images'] = $mo->zeros([50000,32,32,3],NDArray::uint8);
        $dataset['train_labels'] = $mo->zeros([50000,1],NDArray::uint8);
        $this->convertDataset(
            $filenames,
            $dataset['train_images'],
            $dataset['train_labels']);
        $dataset['test_images'] = $mo->zeros([10000,32,32,3],NDArray::uint8);
        $dataset['test_labels'] = $mo->zeros([10000,1],NDArray::uint8);
        $this->convertDataset(
            $testFiles,
            $dataset['test_images'],
            $dataset['test_labels']);
        return $dataset;
    }

    protected function convertDataset($filenames, $image_dataset, $labels_dataset)
    {
        $offset = 0;
        foreach($filenames as $filename) {
            $images = $image_dataset[[$offset,$offset+9999]];
            $labels = $labels_dataset[[$offset,$offset+9999]];
            $this->convertImage(
                $filename, $images, $labels
            );
            $offset += 10000;
        }
    }

    protected function convertImage($filename,$images,$labels)
    {
        $mo = $this->matrixOperator;
        $filePath = $this->datasetDir."/".$filename;
        $imageSize = array_product($this->imageShape);

        $this->console("Converting ".$filename." to NDArray ...");
        $p = 0;
        $f = fopen($filename,'rb');
        $j=0;
        while(true){
            $label = fread($f,1);
            if($label===false)
                break;
            $this->unpackLabel(
                $label,
                $labels[$p]->buffer());
            $red = fread($f,1024);
            $green = fread($f,1024);
            $blue = fread($f,1024);
            if($red===false||
                $green===false||
                $blue===false)
                break;
            $this->unpackImage(
                $red,$green,$blue,
                $images[$p]->buffer());
            $p++;
            $j++;
            if($j>=200) {
                $j=0;
                $this->console('.');
            }
        }
        fclose($f);

        $this->console("Done\n");
        return $data;
    }

    protected function unpackLabel(
        $data,$buffer)
    {
        $values = unpack("C*",$data);
        $i=0;
        foreach ($values as $value) {
            $buffer[$i] = $value;
            $i++;
        }
    }
    protected function unpackImage(
        $reddata,$greendata,$bluedata,$buffer)
    {
        $red = unpack("C*",$reddata);
        $green = unpack("C*",$reddata);
        $blue = unpack("C*",$reddata);
        $size = count($red);
        for($i=0,$j=0;$i<$size;$i++) {
            $buffer[$j++] = $red[$i];
            $buffer[$j++] = $green[$i];
            $buffer[$j++] = $blue[$i];
        }
    }
}
