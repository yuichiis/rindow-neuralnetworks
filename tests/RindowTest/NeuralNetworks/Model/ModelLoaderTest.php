<?php
namespace RindowTest\NeuralNetworks\Model\ModelLoaderTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\Math\Plot\Plot;
use Rindow\Math\Plot\Renderer\GDDriver;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Model\ModelLoader;
use PDO;
use Interop\Polite\Math\Matrix\NDArray;

class Test extends TestCase
{
    private $plot=true;
    private $filename;

    public function newBackend($mo)
    {
        $builder = new NeuralNetworks($mo);
        return $builder->backend();
    }

    public function getPlotConfig()
    {
        return [
            'renderer.skipCleaning' => true,
            'renderer.skipRunViewer' => getenv('TRAVIS_PHP_VERSION') ? true : false,
        ];
    }

    public function setUp() : void
    {
        $this->filename = __DIR__.'/../../../tmp/savedmodel.hda.sqlite3';
        $pdo = new PDO('sqlite:'.$this->filename);
        $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
        $sql = "DROP TABLE IF EXISTS hda";
        $stat = $pdo->exec($sql);
        unset($stat);
        unset($pdo);
    }

    public function testCleanUp()
    {
        $renderer = new GDDriver();
        $renderer->cleanUp();
        $this->assertTrue(true);
    }

    public function testModelFromConfig()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $loader = new ModelLoader($backend,$nn);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2]),
            $nn->layers()->BatchNormalization(),
            $nn->layers()->Activation('sigmoid'),
            $nn->layers()->Dense($units=2, activation:'softmax'),
        ]);
        $model->compile();
        $json = $model->toJson();
        $config = json_decode($json,true);


        // load model
        $model = $loader->modelFromConfig($config);
        $this->assertEquals($json,$model->toJson());

        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1]);
        $history = $model->fit($x,$t, epochs:100, verbose:0);

        $y = $model->predict($x);
        $this->assertEquals($t->toArray(),$mo->argMax($y,$axis=1)->toArray());
    }

    public function testSaveAndLoadModelDefaultDenseBatchNrm()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2]),
            $nn->layers()->BatchNormalization(),
            $nn->layers()->Activation('sigmoid'),
            $nn->layers()->Dense($units=2, activation:'softmax'),
        ]);
        $model->compile();
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1]);
        $history = $model->fit($x,$t,epochs:100, verbose:0);
        [$loss,$accuracy] = $model->evaluate($x,$t);
        $y = $model->predict($x);

        $model->save($this->filename);

        // load model
        $model = $nn->models()->loadModel($this->filename);

        [$loss2,$accuracy2] = $model->evaluate($x,$t);
        $this->assertLessThan(0.5,abs($loss-$loss2));
        $this->assertLessThan(0.5,abs($accuracy-$accuracy2));
        $y2 = $model->predict($x);
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($y,'-',$y2))));
        //$this->assertEquals($t->toArray(),$mo->argMax($y,$axis=1)->toArray());
    }

    public function testSaveAndLoadModelDefaultRnnEmbed()
    {
        $mo = new MatrixOperator();
        $backend = $K = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);

        $REVERSE = True;
        $WORD_VECTOR = 16;
        $UNITS = 128;
        $question = $mo->array([
            [1,2,3,4,5,6],
            [3,4,5,6,7,8],
            [6,5,4,3,2,1],
            [8,7,6,5,4,3],
        ],NDArray::int32);
        $answer = $mo->array([
            [2,4,6],
            [4,6,8],
            [5,3,1],
            [7,5,3],
        ],NDArray::int32);
        $input_length = $question->shape()[1];
        $input_dict_size = $mo->max($question)+1;
        $output_length = $answer->shape()[1];
        $target_dict_size = $mo->max($answer)+1;

        $model = $nn->models()->Sequential([
            $nn->layers()->Embedding($input_dict_size, $WORD_VECTOR,
                input_length:$input_length
            ),
            # Encoder
            $nn->layers()->GRU($UNITS,
                go_backwards:$REVERSE,
                #reset_after:false,
            ),
            # Expand to answer length and peeking hidden states
            $nn->layers()->RepeatVector($output_length),
            # Decoder
            $nn->layers()->GRU($UNITS,
                return_sequences:true,
                go_backwards:$REVERSE,
                #reset_after:false,
            ),
            # Output
            $nn->layers()->Dense(
                $target_dict_size,
                activation:'softmax'
            ),
        ]);
        $model->compile(
            loss:'sparse_categorical_crossentropy',
            optimizer:'adam',
        );
        $history = $model->fit($question,$answer,epochs:10, verbose:0);
        [$loss,$accuracy] = $model->evaluate($question,$answer);
        $y = $model->predict($question);
        $layers = $model->layers();
        $embvals = $layers[0]->getParams();
        $gruvals = $layers[1]->getParams();
        $gru2vals = $layers[3]->getParams();
        $densevals = $layers[4]->getParams();

        $model->save($this->filename);

        // load model
        $model = $nn->models()->loadModel($this->filename);

        [$loss2,$accuracy2] = $model->evaluate($question,$answer);
        $this->assertLessThan(0.5,abs($loss-$loss2));
        $this->assertLessThan(0.5,abs($accuracy-$accuracy2));

        $layers1 = $model->layers();
        $embvals1 = $layers1[0]->getParams();
        $gruvals1 = $layers1[1]->getParams();
        $gru2vals1 = $layers1[3]->getParams();
        $densevals1 = $layers1[4]->getParams();

        $y2 = $model->predict($question);
        $layers2 = $model->layers();
        $embvals2 = $layers2[0]->getParams();
        $gruvals2 = $layers2[1]->getParams();
        $gru2vals2 = $layers2[3]->getParams();
        $densevals2 = $layers2[4]->getParams();
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($embvals[0],'-',$embvals2[0]))));
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($gruvals[0],'-',$gruvals2[0]))));
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($gruvals[1],'-',$gruvals2[1]))));
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($gruvals[2],'-',$gruvals2[2]))));
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($gru2vals[0],'-',$gru2vals2[0]))));
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($gru2vals[1],'-',$gru2vals2[1]))));
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($gru2vals[2],'-',$gru2vals2[2]))));
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($densevals[0],'-',$densevals2[0]))));
        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($densevals[1],'-',$densevals2[1]))));

        $this->assertNotEquals(spl_object_id($embvals[0]),spl_object_id($embvals2[0]));
        $this->assertNotEquals(spl_object_id($gruvals[0]),spl_object_id($gruvals2[0]));
        $this->assertNotEquals(spl_object_id($gruvals[1]),spl_object_id($gruvals2[1]));
        $this->assertNotEquals(spl_object_id($gruvals[2]),spl_object_id($gruvals2[2]));
        $this->assertNotEquals(spl_object_id($gru2vals[0]),spl_object_id($gru2vals2[0]));
        $this->assertNotEquals(spl_object_id($gru2vals[1]),spl_object_id($gru2vals2[1]));
        $this->assertNotEquals(spl_object_id($gru2vals[2]),spl_object_id($gru2vals2[2]));
        $this->assertNotEquals(spl_object_id($densevals[0]),spl_object_id($densevals2[0]));
        $this->assertNotEquals(spl_object_id($densevals[1]),spl_object_id($densevals2[1]));

        $this->assertEquals(spl_object_id($embvals1[0]),spl_object_id($embvals2[0]));
        $this->assertEquals(spl_object_id($gruvals1[0]),spl_object_id($gruvals2[0]));
        $this->assertEquals(spl_object_id($gruvals1[1]),spl_object_id($gruvals2[1]));
        $this->assertEquals(spl_object_id($gruvals1[2]),spl_object_id($gruvals2[2]));
        $this->assertEquals(spl_object_id($gru2vals1[0]),spl_object_id($gru2vals2[0]));
        $this->assertEquals(spl_object_id($gru2vals1[1]),spl_object_id($gru2vals2[1]));
        $this->assertEquals(spl_object_id($gru2vals1[2]),spl_object_id($gru2vals2[2]));
        $this->assertEquals(spl_object_id($densevals1[0]),spl_object_id($densevals2[0]));
        $this->assertEquals(spl_object_id($densevals1[1]),spl_object_id($densevals2[1]));

        $this->assertLessThan(1e-7,$mo->la()->sum($mo->la()->square($mo->op($y,'-',$y2))));
        //$this->assertEquals($t->toArray(),$mo->argMax($y,$axis=1)->toArray());
    }

    public function testSaveAndLoadModelPortable()
    {
        $mo = new MatrixOperator();
        $backend = $this->newBackend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $plt = new Plot($this->getPlotConfig(),$mo);

        $model = $nn->models()->Sequential([
            $nn->layers()->Dense($units=128,input_shape:[2]),
            $nn->layers()->BatchNormalization(),
            $nn->layers()->Activation('sigmoid'),
            $nn->layers()->Dense($units=2, activation:'softmax'),
        ]);
        $model->compile();
        $x = $mo->array([[1, 3], [1, 4], [2, 4], [3, 1], [4, 1], [4, 2]]);
        $t = $mo->array([0, 0, 0, 1, 1, 1]);
        $history = $model->fit($x,$t, epochs:100, verbose:0);
        $y = $model->predict($x);
        [$loss,$accuracy] = $model->evaluate($x,$t);

        $model->save($this->filename,$portable=true);

        // load model
        $model = $nn->models()->loadModel($this->filename);

        $z = $model->predict($x);
        if($this->plot) {
            [$fig,$ax] = $plt->subplots(2);
            $diff = $mo->f('abs',$mo->select($mo->op($y,'-',$z),$mo->arange($t->size()),$mo->zeros([$t->size()])));
            $ax[0]->bar($mo->arange($diff->size()),$diff,null,null,'difference');
            $ax[0]->legend();
            $ax[1]->plot($mo->array($history['loss']),null,null,'loss');
            $ax[1]->plot($mo->array($history['accuracy']),null,null,'accuracy');
            $ax[1]->legend();
            $plt->title('save portable mode');
            $plt->show();
        }

        [$loss2,$accuracy2] = $model->evaluate($x,$t);
        $this->assertLessThan(0.5,abs($loss-$loss2));
        $this->assertLessThan(0.5,abs($accuracy-$accuracy2));
        //$this->assertEquals($t->toArray(),$mo->argMax($y,$axis=1)->toArray());
    }
}
