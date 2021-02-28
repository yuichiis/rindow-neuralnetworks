<?php
namespace RindowTest\NeuralNetworks\Data\Sequence\TextClassifiedDatasetTest;

use PHPUnit\Framework\TestCase;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Data\Sequence\TextClassifiedDataset;

class Test extends TestCase
{
    public function testNormal()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $dataset = new TextClassifiedDataset(
            $mo,
            __DIR__.'/../Dataset/text',
            [
                'pattern'=>'@.*\\.txt@',
                'batch_size'=>2,
            ]
        );
        $dataset->fitOnTexts();
        $this->assertEquals(5,$dataset->datasetSize());

        // sequential access
        $datas = [];
        $sets = [];
        foreach ($dataset as $key => $value) {
            $datas[] = $value;
            [$texts,$labels] = $value;
            foreach($texts as $key => $text) {
                $label = $labels[$key];
                $sets[] = [$text,$label];
            }
        }
        $this->assertCount(3,$datas);
        $this->assertCount(5,$sets);
        $this->assertEquals(3,count($dataset));
        $this->assertEquals(5,$dataset->datasetSize());
        $this->assertInstanceof(NDArray::class,$datas[0][0]);
        //$this->assertEquals([2,3],$datas[0][0]->shape());

        $tokenizer = $dataset->getTokenizer();
        $this->assertEquals(10,$tokenizer->numWords());
        $textDatas = $tokenizer->sequencesToTexts($datas[0][0]);
        $this->assertEquals('negative0 comment text',$textDatas[0]);
        // epoch 2
        $datas = [];
        foreach ($dataset as $key => $value) {
            $datas[] = $value;
        }
        $this->assertCount(3,$datas);

        $this->assertEquals(['neg','pos'],$dataset->classnames());
        //public function loadData(string $filePath=null)

        [$inputs,$tests] = $dataset->loadData();
        $this->assertInstanceof(NDArray::class,$inputs);
        $this->assertInstanceof(NDArray::class,$tests);
        $this->assertEquals([5,4],$inputs->shape());
        $this->assertEquals([5],$tests->shape());
        $txts = $tokenizer->sequencesToTexts($inputs);
        $this->assertCount(5,$txts);
        $results = [];
        $testResults = [];
        foreach ($txts as $key => $txt) {
            $results[$key] = $txt;
        }
        asort($results);
        foreach ($results as $key => $txt) {
            $testResults[] = $tests[$key];
        }
        $results = array_values($results);
        $this->assertEquals([
            "negative0 comment text",
            "negative1 text",
            "positive0 message text",
            "positive1 some message text",
            "positive2 text",
        ],$results);
        $this->assertEquals([0,0,1,1,1],$testResults);
    }

    public function testJustloaddata()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $dataset = new TextClassifiedDataset(
            $mo,
            __DIR__.'/../Dataset/text',
            [
                'pattern'=>'@.*\\.txt@',
                'batch_size'=>2,
                //'verbose'=>1,
            ]
        );

        [$inputs,$tests] = $dataset->loadData();
        $tokenizer = $dataset->getTokenizer();
        $this->assertInstanceof(NDArray::class,$inputs);
        $this->assertInstanceof(NDArray::class,$tests);
        $this->assertEquals([5,4],$inputs->shape());
        $this->assertEquals([5],$tests->shape());
        $txts = $tokenizer->sequencesToTexts($inputs);
        $this->assertEquals("negative0 comment text",$txts[0]);
        $this->assertEquals("negative1 text",$txts[1]);
        $this->assertEquals("positive0 message text",$txts[2]);
        $this->assertEquals("positive1 some message text",$txts[3]);
        $this->assertEquals("positive2 text",$txts[4]);
    }

    public function testLoadValidationData()
    {
        $mo = new MatrixOperator();
        $nn = new NeuralNetworks($mo);

        $dataset = new TextClassifiedDataset(
            $mo,
            __DIR__.'/../Dataset/text',
            [
                'pattern'=>'@.*\\.txt@',
                'batch_size'=>2,
                //'verbose'=>1,
            ]
        );

        [$inputs,$tests] = $dataset->loadData();
        $tokenizer = $dataset->getTokenizer();
        $labels = $dataset->getLabels();
        $val_dataset = new TextClassifiedDataset(
            $mo,
            __DIR__.'/../Dataset/text',
            [
                'pattern'=>'@.*\\.txt@',
                'batch_size'=>2,
                //'verbose'=>1,
                'tokenizer'=>$tokenizer,
                'labels'=>$labels,
            ]
        );

        [$val_inputs,$val_tests] = $val_dataset->loadData();

        $this->assertInstanceof(NDArray::class,$val_inputs);
        $this->assertInstanceof(NDArray::class,$val_tests);
        $this->assertEquals([5,4],$val_inputs->shape());
        $this->assertEquals([5],$val_tests->shape());
        $txts = $tokenizer->sequencesToTexts($val_inputs);
        $this->assertEquals("negative0 comment text",$txts[0]);
        $this->assertEquals("negative1 text",$txts[1]);
        $this->assertEquals("positive0 message text",$txts[2]);
        $this->assertEquals("positive1 some message text",$txts[3]);
        $this->assertEquals("positive2 text",$txts[4]);
    }
}
