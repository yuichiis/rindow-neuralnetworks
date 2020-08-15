<?php
namespace RindowTest\NeuralNetworks\Model\CustomModelTest;

use PHPUnit\Framework\TestCase;
use Rindow\Math\Matrix\MatrixOperator;
use Rindow\NeuralNetworks\Backend\RindowBlas\Backend;
use Rindow\NeuralNetworks\Builder\NeuralNetworks;
use Rindow\NeuralNetworks\Model\ModelLoader;
use Rindow\NeuralNetworks\Model\AbstractModel;
use Interop\Polite\Math\Matrix\NDArray;

class TestModel extends AbstractModel
{
    protected $flat;
    protected $custom;
    protected $fc;
    
    public function __construct($backend,$builder,$hda)
    {
        parent::__construct(
            $backend,
            $builder,
            $builder->utils()->HDA());
        $this->flatten = $builder->Flatten(['input_shape'=>[5]]);
        $this->custom = new TestLayer($backend,$builder);
        $this->fc = $builder->Dense(
            10,
            ['activation'=>'softmax']
        );
    }
    
    protected function buildLayers(array $options=null) : void
    {
        $shape = $this->registerLayer($this->flatten);
        $shape = $this->registerLayer($this->custom,$shape);
        $shape = $this->registerLayer($this->fc,$shape);
    }

    protected function forwardStep(NDArray $inputs, NDArray $trues=null, bool $training=null) : NDArray
    {
        $flat = $this->flat->forward($inputs,$training);
        $customout = $this->custom->forward($flat,$training);
        $outputs = $this->fc->forward($customout,$training);
        return $outputs;
    }
    
    protected function backwardStep(NDArray $dout) : NDArray
    {
        $dout = $this->fc->backward($dout);
        $dout = $this->custom->backward($dout);
        $dInputs = $this->flat->backward($dout);
        return $dInputs;
    }
}

class TestLayer
{
    public function __construct($backend,$builder)
    {
        $this->backend = $backend;
        $this->fc = $builder->Dense(5);
    }
    
    public function build(array $inputShape=null, array $options=null) : array
    {
        $inputShape=$this->normalizeInputShapetShape($inputShape);
        $shape = $this->registerLayer($this->fc,$inputShape);
        $this->outputShape = $shape;
        return $this->outputShape;
    }

    protected function call(NDArray $inputs,bool $training)
    {
        $out = $this->fc->forward($inputs,$training);
        return $out;
    }
    
    protected function differentiate(NDArray $dOutputs)
    {
        $din = $this->fc->backend($dOutputs,$training);
        return $din;
    }
}

class Test extends TestCase
{
    public function testComplieAndFit()
    {
        $mo = new MatrixOperator();
        $backend = new Backend($mo);
        $nn = new NeuralNetworks($mo,$backend);
        $model = new TestModel($backend,$nn);
        
        $train = $mo->random()->randn([10,5]);
        $label = $mo->arange(10);
        $val_train = $mo->random()->randn([10,5]);
        $val_label = $mo->arange(10);
        
        $model->compile();
        $history = $model->fit(
            $train,$label,
            ['epochs'=>5,'batch_size'=>2,'validation_data'=>[$val_train,$val_label]]
        );
        $this->assertTrue(true);
    }
}
