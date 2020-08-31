<?php
namespace Rindow\NeuralNetworks\Layer;

use InvalidArgumentException;
use Interop\Polite\Math\Matrix\NDArray;
use Rindow\NeuralNetworks\Support\GenericUtils;

class GRUCell extends AbstractRNNCell 
{
    use GenericUtils;
    protected $backend;
    protected $units;
    protected $useBias;
    protected $kernelInitializer;
    protected $biasInitializer;
    protected $ac;
    protected $ac_i;
    protected $ac_f;
    protected $ac_c;
    protected $ac_o;

    protected $kernel;
    protected $recurrentKernel;
    protected $bias;
    protected $dKernel;
    protected $dRecurrentKernel;
    protected $dBias;
    protected $inputs;

    public function __construct($backend,int $units, array $options=null)
    {
        extract($this->extractArgs([
            'input_shape'=>null,
            'activation'=>'tanh',
            'recurrent_activation'=>'sigmoid',
            'use_bias'=>true,
            'kernel_initializer'=>'sigmoid_normal',
            'recurrent_initializer'=>'sigmoid_normal',
            'bias_initializer'=>'zeros',
            'unit_forget_bias'=>true,
            //'kernel_regularizer'=>null, 'bias_regularizer'=>null,
            //'activity_regularizer'=null,
            //'kernel_constraint'=null, 'bias_constraint'=null,
        ],$options));
        $this->backend = $K = $backend;
        $this->units = $units;
        $this->inputShape = $input_shape;
        if($use_bias) {
            $this->useBias = $use_bias;
        }
        $this->ac_hh = $this->createFunction($activation);
        $this->ac_z = $this->createFunction($recurrent_activation);
        $this->ac_r = $this->createFunction($recurrent_activation);
        $this->kernelInitializer = $K->getInitializer($kernel_initializer);
        $this->recurrentInitializer = $K->getInitializer($recurrent_initializer);
        $this->biasInitializer   = $K->getInitializer($bias_initializer);
        $this->kernelInitializerName = $kernel_initializer;
        $this->recurrentInitializerName = $recurrent_initializer;
        $this->biasInitializerName = $bias_initializer;
    }

    public function build(array $inputShape=null, array $options=null) : array
    {
        extract($this->extractArgs([
            'sampleWeights'=>null,
        ],$options));
        $K = $this->backend;
        $kernelInitializer = $this->kernelInitializer;
        $recurrentInitializer = $this->recurrentInitializer;
        $biasInitializer = $this->biasInitializer;

        $inputShape = $this->normalizeInputShape($inputShape);
        //if(count($inputShape)!=1) {
        //    throw new InvalidArgumentException(
        ///        'Unsuppored input shape: ['.implode(',',$inputShape).']');
        //}
        $shape = $inputShape;
        $inputDim = array_pop($shape);
        if($sampleWeights) {
            $this->kernel = $sampleWeights[0];
            $this->recurrentKernel = $sampleWeights[1];
            $this->bias = $sampleWeights[2];
        } else {
            $this->kernel = $kernelInitializer([$inputDim,$this->units*3],$inputDim);
            $this->recurrentKernel = $recurrentInitializer([$this->units*3,$this->units],$this->units*3);
            if($this->useBias) {
                $this->bias = $biasInitializer([$this->units*3]);
            }
        }
        $this->dKernel = $K->zerosLike($this->kernel);
        $this->dRecurrentKernel = $K->zerosLike($this->recurrentKernel);
        if($this->bias) {
            $this->dBias = $K->zerosLike($this->bias);
        }
        $this->r_kernel_z = $this->recurrentKernel[[0,$this->units-1]];
        $this->r_kernel_r = $this->recurrentKernel[[$this->units,$this->units*2-1]];
        $this->r_kernel_hh = $this->recurrentKernel[[$this->units*2,$this->units*3-1]];
        $this->dR_kernel_z = $this->dRecurrentKernel[[0,$this->units-1]];
        $this->dR_kernel_r = $this->dRecurrentKernel[[$this->units,$this->units*2-1]];
        $this->dR_kernel_hh = $this->dRecurrentKernel[[$this->units*2,$this->units*3-1]];
        array_push($shape,$this->units);
        $this->outputShape = $shape;
        return $this->outputShape;
    }

    public function getParams() : array
    {
        if($this->bias) {
            return [$this->kernel,$this->recurrentKernel,$this->bias];
        } else {
            return [$this->kernel,$this->recurrentKernel];
        }
    }

    public function getGrads() : array
    {
        if($this->bias) {
            return [$this->dKernel,$this->dRecurrentKernel,$this->dBias];
        } else {
            return [$this->dKernel,$this->dRecurrentKernel];
        }
    }

    public function getConfig() : array
    {
        return [
            'units' => $this->units,
            'options' => [
                'input_shape'=>$this->inputShape,
                'use_bias'=>$this->useBias,
                'activation'=>$this->activationName,
                'recurrent_activation'=>$this->recurrentActivationName,
                'kernel_initializer' => $this->kernelInitializerName,
                'recurrent_initializer' => $this->recurrentInitializerName,
                'bias_initializer' => $this->biasInitializerName,
            ]
        ];
    }

    protected function call(NDArray $inputs, array $states, bool $training, object $calcState, array $options=null) : array
    {
        $K = $this->backend;
        $prev_h = $states[0];
        
        if($this->bias){
            $gateOuts = $K->batch_gemm($inputs, $this->kernel,1.0,1.0,$this->bias);
        } else {
            $gateOuts = $K->gemm($inputs, $this->kernel);
        }
        $x_z = $K->slice($gateOuts,
            [0,0],[-1,$this->units]);
        $x_r = $K->slice($gateOuts,
            [0,$this->units],[-1,$this->units]);
        $x_hh = $K->slice($gateOuts,
            [0,$this->units*2],[-1,$this->units]);
        
        $x_z = $K->gemm($prev_h, $this->r_kernel_z,1.0,1.0,$x_z);
        $x_r = $K->gemm($prev_h, $this->r_kernel_r,1.0,1.0,$x_r);
        $x_hh = $K->gemm($K->mul($x_r,$prev_h), $this->r_kernel_hh,1.0,1.0,$x_hh);

        if($this->ac_z){
            $x_z = $this->ac_z->call($x_z,$training);
            $x_r = $this->ac_r->call($x_r,$training);
        }
        if($this->ac_hh){
            $x_hh = $this->ac_hh->call($x_hh,$training);
        }
        
        // next_h = (1-z) * prev_h + z * hh
        $x1_z = $K->increment($K->scale(-1,$x_z),1);
        $next_h = $K->add(
            $K->mul($x1_z,$prev_h),
            $K->mul($x_z,$x_hh));
        
        $calcState->inputs = $inputs;
        $calcState->prev_h = $prev_h;
        $calcState->x_z = $x_z;
        $calcState->x1_z = $x1_z;
        $calcState->x_r = $x_r;
        $calcState->x_hh = $x_hh;
        
        return [$next_h,[$next_h]];
    }

    protected function differentiate(NDArray $dOutputs, array $dStates, object $calcState) : array
    {
        $K = $this->backend;
        $dNext_h = $dStates[0];
        $dNext_h = $K->add($dOutputs,$dNext_h);
        
        // dprev_h = dnext_h * (1-z)
        $dPrev_h = $K->mul($dNext_h,$calcState->x1_z);
        
        // hh output
        $dX_hh = $K->mul($dNext_h,$calcState->x_z);
        if($this->ac_hh){
            $dX_hh = $this->ac_hh->differentiate($dX_hh);
        }
        $K->gemm($K->mul($calcState->x_r, $calcState->prev_h), $dX_hh,1.0,1.0,$this->dR_kernel_hh,true,false);
        $dhh_r = $K->gemm($dX_hh, $this->r_kernel_hh,1.0,0.0,null,false,true);
        $K->update_add($dPrev_h,$K->mul($calcState->x_r,$dhh_r));

        // z gate
        $dX_z = $K->mul($dNext_h,$K->sub($calcState->x_hh,$calcState->prev_h));
        if($this->ac_z){
            $dX_z = $this->ac_z->differentiate($dX_z);
        }
        $K->gemm($calcState->prev_h, $dX_z,1.0,1.0,$this->dR_kernel_z,true,false);
        $K->gemm($dX_z, $this->r_kernel_z,1.0,1.0,$dPrev_h,false,true);

        // r gate
        $dX_r = $K->mul($dhh_r,$calcState->prev_h);
        if($this->ac_r){
            $dX_r = $this->ac_r->differentiate($dX_r);
        }
        $K->gemm($calcState->prev_h, $dX_r,1.0,1.0,$this->dR_kernel_r,true,false);
        $K->gemm($dX_r, $this->r_kernel_r,1.0,1.0,$dPrev_h,false,true);
        
        // stack diff gate outputs
        $dGateOuts = $K->stack(
            [$dX_z,$dX_r,$dX_hh],$axis=1);
        $shape = $dGateOuts->shape();
        $batches = array_shift($shape);
        $dGateOuts = $dGateOuts->reshape([
                $batches,
                array_product($shape)
            ]);

        $K->copy($K->sum($dGateOuts, $axis=0),$this->dBias);
        $K->gemm($calcState->inputs, $dGateOuts,1.0,1.0,$this->dKernel,true,false);
        $dInputs = $K->gemm($dGateOuts, $this->kernel,1.0,1.0,null,false,true);

var_dump($this->dR_kernel_z->toArray());
var_dump($this->dR_kernel_r->toArray());
var_dump($this->dR_kernel_hh->toArray());
var_dump($this->dR_kernel_z->toArray());
        return [$dInputs,[$dPrev_h]];
    }
}
